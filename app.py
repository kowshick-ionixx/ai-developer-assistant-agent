"""
app.py
------
This file is ONLY responsible for the Streamlit user interface: showing the
chat, taking user input, and displaying the agent's answers. All the AI logic
lives in agent.py, and the tools live in tools.py.

    USER -> STREAMLIT UI -> agent.py (LangChain + Gemini) -> back to UI
"""

import html
import os
import re
import time
import uuid
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

import documents
import tools
import workflow
from agent import (
    AGENT_TEMPERATURE,
    build_agent,
    get_api_key,
    get_model_name,
    new_ai_message,
    new_human_message,
    run_agent_turn,
    transcribe_audio,
)
from logger import clear_events, get_recent_events, log_error

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Developer Assistant",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .tool-box {
        background-color: rgba(120, 120, 120, 0.08);
        border-left: 4px solid #4CAF50;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }

    /* Chat responses are rendered from LLM-generated Markdown, which can
       contain ATX headings (#, ##, ...) copied verbatim from scraped
       documentation/web content. Markdown headings are allowed to interrupt
       a paragraph, so a single stray "# ..." line (e.g. a shell comment)
       renders as a full browser-default heading and breaks typography
       consistency. These rules clamp every heading level to one of two
       small, bold sizes and pin body/list/code text to fixed sizes, using
       rem/em units only (no colors) so both the light and dark themes are
       unaffected.
    */
    [data-testid="stChatMessage"] :is(h1, h2, h3, h4, h5, h6) {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        margin: 0.6rem 0 0.3rem !important;
        line-height: 1.4 !important;
    }
    [data-testid="stChatMessage"] :is(h1, h2) {
        font-size: 1.15rem !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        font-size: 1rem !important;
        line-height: 1.55 !important;
    }
    [data-testid="stChatMessage"] code {
        font-size: 0.85em !important;
    }

    /* Defensive guard: force full brightness on the main content area
       regardless of Streamlit's internal script-run state or any external
       interference (e.g. a browser extension), so the page can never be
       left dimmed - during processing or after a response completes. */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .block-container,
    [data-testid="stChatMessage"] {
        opacity: 1 !important;
        filter: none !important;
    }

    /* ---------------------------------------------------------------
       Dashboard-style panels (Phase checklist, stats, workflow trail,
       tool chips, diffs, terminal output, final report banner). Pure
       presentation on top of real session data - see the render_* /
       compute_* helpers below for what actually feeds each class.
       --------------------------------------------------------------- */
    section[data-testid="stSidebar"] { border-right: 1px solid #262635; }

    .panel-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid #262635;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.8rem;
    }
    .panel-title {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #9a9ab0;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .phase-checklist { display: flex; flex-direction: column; gap: 2px; }
    .phase-row {
        display: flex; align-items: center; gap: 8px;
        padding: 6px 10px; border-radius: 8px;
        font-size: 0.85rem; color: #c9c9d6;
    }
    .phase-row.active {
        background: rgba(139,92,246,0.18);
        border: 1px solid rgba(139,92,246,0.5);
        color: #fff; font-weight: 600;
    }
    .phase-check { color: #22c55e; font-weight: 700; }
    .phase-label { color: #8a8a9c; margin-left: auto; font-size: 0.78rem; }
    .phase-row.active .phase-label { color: #d8cbfd; }

    .stat-row {
        display: flex; justify-content: space-between;
        padding: 4px 0; font-size: 0.85rem; color: #c9c9d6;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-value { color: #fff; font-weight: 700; }

    .wf-list { display: flex; flex-direction: column; gap: 2px; }
    .wf-row { padding: 5px 10px; border-radius: 6px; font-size: 0.83rem; }
    .wf-row.done { color: #22c55e; }
    .wf-row.active { color: #a78bfa; background: rgba(139,92,246,0.14); font-weight: 700; }
    .wf-row.pending { color: #62626f; }

    .tool-chip-list { display: flex; flex-direction: column; gap: 4px; }
    .tool-chip {
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.3);
        color: #4ade80; padding: 4px 10px; border-radius: 6px;
        font-size: 0.78rem;
    }

    .diff-block, .terminal-block {
        background: #0a0a10; border: 1px solid #262635; border-radius: 8px;
        padding: 10px; font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.78rem; white-space: pre-wrap; overflow-x: auto;
        max-height: 340px; overflow-y: auto; margin: 0;
    }
    .diff-add { color: #4ade80; display: block; }
    .diff-del { color: #f87171; display: block; }
    .diff-ctx { color: #9a9ab0; display: block; }
    .terminal-block { color: #4ade80; }
    .terminal-block.error { color: #f87171; }

    .test-banner {
        border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; font-size: 0.9rem;
    }
    .test-banner.pass {
        background: rgba(34,197,94,0.12); color: #4ade80;
        border: 1px solid rgba(34,197,94,0.35);
    }
    .test-banner.fail {
        background: rgba(239,68,68,0.12); color: #f87171;
        border: 1px solid rgba(239,68,68,0.35);
    }
    .test-row { font-size: 0.82rem; padding: 2px 0; color: #c9c9d6; }

    .final-report-banner {
        margin-top: 1rem;
        background: linear-gradient(90deg, rgba(34,197,94,0.16), rgba(34,197,94,0.04));
        border: 1px solid rgba(34,197,94,0.4);
        border-radius: 12px; padding: 16px 20px;
    }
    .final-report-title { font-size: 1.05rem; font-weight: 700; color: #4ade80; margin-bottom: 10px; }
    .final-report-stats { display: flex; gap: 28px; flex-wrap: wrap; }
    .final-report-stats > div { display: flex; flex-direction: column; font-size: 0.78rem; color: #9a9ab0; }
    .final-report-stats > div b { font-size: 1rem; color: #fff; margin-top: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# AI Features reference panel (sidebar): pure information about what each
# phase can do - no example question is ever inserted into the chat input or
# sent to the agent from here. The user always types their own question into
# the existing chat bar; this dict only feeds read-only st.markdown() text.
AI_FEATURES_INFO = {
    "Phase 1 — Foundation & Basic Agent": {
        "summary": "Provides the basic AI agent foundation.",
        "features": [
            "Basic Gemini/LangChain AI agent",
            "Chat interaction",
            "Conversation history",
            "Calculator",
            "Python explanation",
            "Basic AI agent behavior",
            "Environment/API key handling",
            "Basic error handling",
            "Scope restriction",
        ],
        "example": "Ask the AI about Python, programming concepts, or perform a calculation.",
    },
    "Phase 2 — Developer Skills": {
        "summary": "Makes the AI useful as a software-development assistant.",
        "features": [
            "Code generation",
            "Code explanation",
            "Debugging",
            "Code review",
            "Refactoring",
            "Code optimization suggestions",
            "Multiple programming languages",
            "Pytest",
            "Ruff",
            "Black",
            "Controlled developer tools",
            "Tool execution logging",
            "Security restrictions",
        ],
        "example": "Generate, explain, debug, review, or refactor code.",
    },
    "Phase 3 — Project / Codebase Understanding": {
        "summary": "Allows the AI to understand the actual project.",
        "features": [
            "List project files",
            "Read project files",
            "Search project code",
            "Find relevant files",
            "Project structure inspection",
            "Project-aware questions",
            "Cross-file reasoning",
            "Focused context retrieval",
            "Safe file access",
            "Secret protection",
            "Path traversal protection",
        ],
        "example": "Ask the AI to explain how files or components in the project work together.",
    },
    "Phase 4 — Web Search, Documentation & Git/GitHub": {
        "summary": (
            "Allows the AI to obtain external technical information and "
            "inspect version-control information."
        ),
        "features": [
            "Web search",
            "Documentation search",
            "Current technical information",
            "Git status",
            "Git log",
            "Git diff",
            "Git branch",
            "GitHub repository information",
            "GitHub issues",
            "GitHub pull requests",
            "Untrusted web-result handling",
            "Git/GitHub security controls",
        ],
        "example": "Ask about current programming documentation or inspect Git/GitHub information.",
    },
    "Phase 5 — Code Execution, Testing & Error Analysis": {
        "summary": (
            "Allows the AI to safely run controlled development operations "
            "and analyze the results."
        ),
        "features": [
            "Run pytest",
            "Test result analysis",
            "Passed/failed test counting",
            "stdout capture",
            "stderr capture",
            "Exit-code handling",
            "Timeout handling",
            "Traceback analysis",
            "Runtime-error analysis",
            "Syntax-error detection",
            "Failure analysis",
            "Regression testing",
            "Ruff validation",
            "Black validation",
            "Controlled execution",
            "Security restrictions",
        ],
        "example": "Run the project tests and explain any failures.",
    },
    "Phase 6 — Autonomous Software Development": {
        "summary": (
            "Allows the AI to carry out multi-step development tasks "
            "(plan, inspect, implement, test, review) with controlled, "
            "human-approved file changes."
        ),
        "features": [
            "Task planning and decomposition",
            "Autonomous tool selection",
            "Controlled file creation/modification",
            "Human approval before any file is written",
            "PDF requirements → project generation",
            "Test → analyze → fix → retest loop (capped retries)",
            "Regression testing",
            "Git-aware development (read-only)",
            "Workflow state tracking",
            "Security restrictions",
        ],
        "example": (
            "Ask the AI to add a feature or fix a bug - it will propose a plan and "
            "file changes, and only apply them once you approve."
        ),
    },
}

TOOL_DISPLAY_NAMES = {
    "calculator": "🔧 Using Calculator Tool",
    "explain_python_code": "🔧 Using Code Explanation Tool",
    "run_pytest": "🔧 Running Pytest",
    "run_ruff": "🔧 Running Ruff",
    "run_black": "🔧 Running Black",
    "check_python_syntax": "🔧 Checking Python Syntax",
    "list_project_files": "🔧 Listing Project Files",
    "read_project_file": "🔧 Reading Project File",
    "search_project": "🔧 Searching Project",
    "web_search": "🌐 Searching the Web",
    "documentation_search": "📚 Searching Documentation",
    "git_status": "🔧 Checking Git Status",
    "git_log": "🔧 Reading Git Log",
    "git_diff": "🔧 Checking Git Diff",
    "git_branch": "🔧 Checking Git Branch",
    "github_get_repository": "🐙 Looking Up GitHub Repository",
    "github_get_issues": "🐙 Fetching GitHub Issues",
    "github_get_pull_requests": "🐙 Fetching GitHub Pull Requests",
    "propose_file_change": "📝 Proposing File Change",
    "apply_approved_change": "✅ Applying Approved Change",
    "list_pending_changes": "🔧 Checking Pending Changes",
}

# "Execution Plan" card: a human-readable label for each tool the agent can
# call, used to render what it actually did this turn as a checklist. This
# is a post-hoc summary of real tool calls (see render_execution_plan) - not
# a forecast, so every line shown already happened and is checked off.
EXECUTION_STEP_LABELS = {
    "list_project_files": "Inspect project structure",
    "search_project": "Search the project for related code",
    "read_project_file": "Read a project file",
    "documentation_search": "Search documentation",
    "web_search": "Search the web",
    "git_status": "Check Git status",
    "git_log": "Read Git log",
    "git_diff": "Check Git diff",
    "git_branch": "Check Git branch",
    "propose_file_change": "Propose a file change",
    "apply_approved_change": "Apply an approved file change",
    "run_pytest": "Run the test suite",
    "run_ruff": "Run Ruff",
    "run_black": "Run Black",
    "check_python_syntax": "Check Python syntax",
    "calculator": "Perform a calculation",
    "explain_python_code": "Analyze code structure",
    "github_get_repository": "Look up the GitHub repository",
    "github_get_issues": "Fetch GitHub issues",
    "github_get_pull_requests": "Fetch GitHub pull requests",
    "list_pending_changes": "Check pending changes",
}

# Quick Actions: an optional prefix prepended to the message sent to the
# agent (the chat still shows the user's original text). "Ask" sends the
# message unchanged - the agent already understands intent on its own.
MODE_PREFIXES = {
    "Ask": "",
    "Generate Code": "Generate code for the following request:\n\n",
    "Debug Code": "Debug the following code and explain the fix:\n\n",
    "Review Code": "Review the following code:\n\n",
    "Refactor Code": "Refactor the following code:\n\n",
    "Generate Tests": "Generate pytest tests for the following code:\n\n",
}

# Sidebar "AI FEATURES (PHASES)" checklist: every phase this codebase has
# actually built (see AI_FEATURES_INFO above for the detailed reference
# panel) is complete - CURRENT_PHASE is only which one is "active" for
# display, not a claim that earlier phases are unfinished.
PHASE_CHECKLIST = [
    ("Phase 1", "Foundation & Basic Agent"),
    ("Phase 2", "Developer Skills"),
    ("Phase 3", "Project Understanding"),
    ("Phase 4", "Knowledge & Git/GitHub"),
    ("Phase 5", "Execution & Testing"),
    ("Phase 6", "Autonomous Agent"),
]
CURRENT_PHASE = "Phase 6"

# The 10-step trail shown in the "Workflow Status" panel. Real Phase 6 turns
# can also pass through a few states not shown as their own row here
# (PROPOSING_CHANGE, WAITING_FOR_APPROVAL, REGRESSION_TESTING, FAILED) -
# those are surfaced instead via the "Current State" badge above the trail,
# never silently folded into one of these rows.
WORKFLOW_ROWS = [
    ("PLANNING", "PLANNING"),
    ("INSPECTING", "INSPECTING"),
    ("SEARCHING_DOCUMENTATION", "SEARCHING DOCS"),
    ("IMPLEMENTING", "IMPLEMENTING"),
    ("TESTING", "TESTING"),
    ("ANALYZING", "ANALYZING"),
    ("FIXING", "FIXING"),
    ("RETESTING", "RETESTING"),
    ("REVIEWING", "REVIEWING"),
    ("COMPLETED", "COMPLETED"),
]

_LANGUAGE_BY_EXTENSION = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "tsx": "tsx",
    "jsx": "jsx",
    "json": "json",
    "md": "markdown",
    "html": "html",
    "css": "css",
    "sh": "bash",
    "yml": "yaml",
    "yaml": "yaml",
    "toml": "toml",
}


def _output_block(text: str, limit: int = 2000) -> str:
    """Render tool output as safely-escaped, pre-formatted HTML."""
    text = str(text).strip()
    if len(text) > limit:
        text = text[:limit] + "\n... (truncated)"
    return f'<pre style="white-space:pre-wrap;margin:0;">{html.escape(text)}</pre>'


# Matches a *complete* ```lang\n...\n``` fenced code block. Only fully
# paired fences are matched - a stray/unclosed ``` is left as plain text
# instead of being treated as an (empty) code block, which is what was
# producing the large empty dark boxes: Markdown's own fence auto-detection
# has no way to tell "malformed fence" from "intentional empty code block".
_CODE_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)


def render_answer(text: str) -> None:
    """Render assistant answer text, drawing fenced code blocks with
    st.code() (guaranteed monospace font + syntax highlighting) instead of
    relying on Markdown's automatic fence detection, and skipping any code
    fence or text segment that has no actual content.
    """
    text = text or ""
    pos = 0
    rendered_anything = False
    for match in _CODE_FENCE_RE.finditer(text):
        before = text[pos : match.start()]
        if before.strip():
            st.markdown(before)
            rendered_anything = True
        language = match.group(1).strip() or None
        code = match.group(2).strip("\n")
        if code.strip():
            st.code(code, language=language)
            rendered_anything = True
        pos = match.end()
    remainder = text[pos:]
    if remainder.strip():
        st.markdown(remainder)
        rendered_anything = True
    if not rendered_anything:
        st.markdown(text)


def _js_string_literal(text: str) -> str:
    """Escape text for safe use inside a JS template literal (backtick string)."""
    return (
        text.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
        .replace("</", "<\\/")
    )


def render_tts_button(text: str, key: str) -> None:
    """Optional 'read aloud' button: uses the browser's own built-in
    speechSynthesis (Web Speech API) client-side - no server-side TTS
    service, API key, or extra dependency involved."""
    if st.button("🔊 Read aloud", key=key):
        components.html(
            f"""
            <script>
            const utterance = new SpeechSynthesisUtterance(`{_js_string_literal(text)}`);
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
            </script>
            """,
            height=0,
        )


def render_workflow_states(states: list[str]) -> None:
    """Show the Phase 6 workflow-status trail for one turn (e.g.
    'Workflow: IMPLEMENTING -> WAITING FOR APPROVAL'), derived from which
    tools were actually called - purely descriptive, shown only when at
    least one Phase 6-relevant tool ran this turn."""
    if not states:
        return
    trail = " → ".join(state.replace("_", " ") for state in states)
    st.caption(f"🔧 Workflow: {trail}")


def render_execution_plan(tool_calls: list[dict]) -> None:
    """'Execution Plan' checklist: what this turn's tool calls actually did,
    in the order they really happened - never a forecast of future steps, so
    every line is already checked off (it already ran). Shown only when the
    turn actually called at least one tool."""
    if not tool_calls:
        return
    steps = [
        EXECUTION_STEP_LABELS.get(tool_call["name"], tool_call["name"])
        for tool_call in tool_calls
    ]
    with st.container(border=True):
        st.markdown("**📋 Execution Plan**")
        for index, step in enumerate(steps, start=1):
            st.markdown(f"{index}. {html.escape(step)} ✅")


def render_tool_box(tool_call: dict) -> None:
    """Render a single '🔧 Using X Tool' box for one tool call."""
    name = tool_call["name"]
    header = TOOL_DISPLAY_NAMES.get(name, f"🔧 Using {name}")

    if name == "calculator":
        expression = tool_call["input"].get("expression", "")
        body = f"**Expression:** `{expression}`<br><br>**Result:** {html.escape(str(tool_call['output']))}"
    elif name == "explain_python_code":
        body = "Analyzed the code's structure without executing it."
    elif name in ("run_ruff", "run_black"):
        target = tool_call["input"].get("file_path") or "pasted code"
        body = f"**Target:** `{target}`<br>{_output_block(tool_call['output'])}"
    elif name == "propose_file_change":
        target = tool_call["input"].get("file_path", "")
        body = f"**File:** `{target}`<br>{_output_block(tool_call['output'])}"
    elif name == "apply_approved_change":
        change_id = tool_call["input"].get("change_id", "")
        body = f"**Change ID:** `{change_id}`<br>{_output_block(tool_call['output'])}"
    else:
        body = _output_block(tool_call["output"])

    st.markdown(
        f'<div class="tool-box"><b>{header}</b><br>{body}</div>', unsafe_allow_html=True
    )


def _run_and_render_turn(
    agent,
    displayed_text: str,
    agent_input: str,
    *,
    source: str | None = None,
    spinner_text: str = "Thinking...",
) -> None:
    """Send `agent_input` through the real agent pipeline (run_agent_turn -
    the same Phase 1-6 agent/workflow this whole app already uses, never a
    second agent or a second workflow engine), render the exchange live, and
    record it in session state exactly like a normal typed chat turn.

    `displayed_text` is what the chat bubble shows the user (may differ from
    `agent_input`, e.g. an approval nudge the agent needs but the user didn't
    literally type) - shared by the main chat_input handler below and by
    _approve_and_resume(), so "typing a message" and "clicking Approve" both
    go through this one real turn-runner instead of duplicating it.
    """
    st.session_state.messages.append(
        {"role": "user", "content": displayed_text, "tool_calls": [], "source": source}
    )
    st.session_state.lc_history.append(new_human_message(agent_input))

    with st.chat_message("user"):
        if source == "voice":
            st.caption("🎤 Voice input · 📝 Transcribed")
        st.markdown(displayed_text)

    with st.chat_message("assistant"):
        turn_started_at = time.time()
        with st.spinner(spinner_text):
            try:
                result = run_agent_turn(agent, st.session_state.lc_history)
                answer = result["answer"]
                tool_calls = result["tool_calls"]
                workflow_states = result.get("workflow_states", [])
            except Exception as exc:  # noqa: BLE001
                print(f"[agent error] {exc}")
                answer = "⚠️ I couldn't process that request. Please check your API key or try again."
                tool_calls = []
                workflow_states = []
        turn_duration = time.time() - turn_started_at

        for tool_call in tool_calls:
            render_tool_box(tool_call)

        render_answer(answer)
        render_workflow_states(workflow_states)
        render_execution_plan(tool_calls)
        render_tts_button(answer, key=f"tts_live_{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "tool_calls": tool_calls,
            "workflow_states": workflow_states,
            "duration_seconds": turn_duration,
        }
    )
    st.session_state.lc_history.append(new_ai_message(answer))


def _approve_and_resume(agent, change: workflow.ProposedChange) -> None:
    """Approve a proposed change via the real Phase 6 approval registry
    (workflow.approve_change - the exact function the sidebar's own Approve
    button already calls), then drive the SAME agent through
    run_agent_turn's existing bounded auto-continuation loop to actually
    apply it, run the tests, and fix/retest within the existing
    repair-attempt limit - i.e. exactly what typing "apply it" would already
    do, just triggered automatically instead of requiring that follow-up
    message."""
    workflow.approve_change(change.change_id)
    nudge = (
        f"The user has approved change '{change.change_id}' "
        f"({change.action} {change.file_path}). Apply it with "
        "apply_approved_change, then run the test suite; if it fails, fix the "
        "issue and retest within the existing repair-attempt limit. Finish "
        "with a short summary of what changed and the final test result."
    )
    displayed = (
        f"✅ Approved change `{change.change_id}` ({change.file_path}). "
        "Please apply it."
    )
    _run_and_render_turn(
        agent, displayed, nudge, spinner_text="Applying the approved change..."
    )


# ---------------------------------------------------------------------------
# Dashboard data helpers - every panel below reads ONLY real session data:
# messages already produced this session (st.session_state.messages, each
# carrying the tool_calls/workflow_states that turn actually produced) and
# workflow.py's real pending/applied-change registry. Nothing here invents
# numbers that weren't actually observed.
# ---------------------------------------------------------------------------


def _last_assistant_message() -> dict | None:
    for message in reversed(st.session_state.messages):
        if message["role"] == "assistant":
            return message
    return None


def _find_tool_calls(messages: list[dict], name: str) -> list[dict]:
    """All tool_calls named `name` across `messages`, oldest first."""
    return [
        tool_call
        for message in messages
        for tool_call in message.get("tool_calls", [])
        if tool_call["name"] == name
    ]


def _applied_changes_this_session() -> list[workflow.ProposedChange]:
    """Changes actually written to disk this session, resolved from the
    real change registry (workflow.get_change keeps a change reachable by
    id even after it's been applied - only list_pending_changes() filters
    those out, since that list is for the *pending* approval queue)."""
    seen_ids: set[str] = set()
    applied: list[workflow.ProposedChange] = []
    for tool_call in _find_tool_calls(
        st.session_state.messages, "apply_approved_change"
    ):
        change_id = tool_call["input"].get("change_id", "")
        if not change_id or change_id in seen_ids:
            continue
        seen_ids.add(change_id)
        change = workflow.get_change(change_id)
        if change is not None and change.applied:
            applied.append(change)
    return applied


def _guess_language(file_path: str) -> str | None:
    if "." not in file_path:
        return None
    return _LANGUAGE_BY_EXTENSION.get(file_path.rsplit(".", 1)[-1].lower())


@st.cache_data(show_spinner=False)
def _get_repo_url() -> str | None:
    """This project's real Git remote URL (read-only), for the header's
    GitHub link - None if it isn't a Git repo or has no remote configured,
    never a fabricated link."""
    try:
        repo = tools._get_repo()
        if repo is None or not repo.remotes:
            return None
        url = repo.remotes[0].url
    except Exception:  # noqa: BLE001 - decorative UI chrome must never crash the page
        return None
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:") :]
    return url.removesuffix(".git")


_PYTEST_TEST_LINE_RE = re.compile(
    r"^(\S+::\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)\b", re.MULTILINE
)
_PYTEST_SUMMARY_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|skipped|errors?)\b")
_PYTEST_DURATION_RE = re.compile(r"\bin\s+([\d.]+)s\b")
_PYTEST_EXIT_CODE_RE = re.compile(r"^Exit code:\s*(\d+)")


def _parse_pytest_output(output: str) -> dict:
    """Pull real counts/durations/per-test results out of run_pytest's own
    output (see tools.py's run_pytest: "Exit code: N\\n\\n<pytest -v output>").
    Never guesses a number it can't find - missing fields stay 0/None."""
    output = str(output)
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for number, label in _PYTEST_SUMMARY_COUNT_RE.findall(output):
        key = "errors" if label.startswith("error") else label
        counts[key] = int(number)
    duration_match = _PYTEST_DURATION_RE.search(output)
    duration = float(duration_match.group(1)) if duration_match else None
    tests = [
        {"name": name, "status": status}
        for name, status in _PYTEST_TEST_LINE_RE.findall(output)
    ]
    exit_code_match = _PYTEST_EXIT_CODE_RE.match(output.strip())
    exit_code = int(exit_code_match.group(1)) if exit_code_match else None
    total = counts["passed"] + counts["failed"] + counts["skipped"] + counts["errors"]
    return {
        "counts": counts,
        "duration": duration,
        "tests": tests,
        "total": total,
        "exit_code": exit_code,
    }


# ---------------------------------------------------------------------------
# Sidebar panel renderers
# ---------------------------------------------------------------------------


def render_phase_checklist() -> None:
    rows = []
    for phase, label in PHASE_CHECKLIST:
        active = phase == CURRENT_PHASE
        row_class = "phase-row active" if active else "phase-row"
        rows.append(
            f'<div class="{row_class}"><span class="phase-check">✓</span>'
            f"<span>{html.escape(phase)}</span>"
            f'<span class="phase-label">{html.escape(label)}</span></div>'
        )
    st.markdown(
        '<div class="panel-card"><div class="panel-title">AI Features (Phases)</div>'
        f'<div class="phase-checklist">{"".join(rows)}</div></div>',
        unsafe_allow_html=True,
    )


def _compute_agent_stats() -> dict:
    messages = st.session_state.messages
    tasks_completed = sum(1 for message in messages if message["role"] == "assistant")
    files_modified = len(
        {change.file_path for change in _applied_changes_this_session()}
    )

    pytest_calls = _find_tool_calls(messages, "run_pytest")
    tests_run = 0
    passing_runs = 0
    for tool_call in pytest_calls:
        parsed = _parse_pytest_output(tool_call["output"])
        tests_run += parsed["total"]
        if parsed["exit_code"] == 0:
            passing_runs += 1
    success_rate = (
        f"{round(passing_runs / len(pytest_calls) * 100)}%" if pytest_calls else "—"
    )

    last_assistant = _last_assistant_message()
    last_states = last_assistant.get("workflow_states", []) if last_assistant else []
    active_workflow = (
        1 if last_states and last_states[-1] not in ("COMPLETED", "FAILED") else 0
    )

    return {
        "Tasks Completed": tasks_completed,
        "Files Modified": files_modified,
        "Tests Run": tests_run,
        "Success Rate": success_rate,
        "Active Workflow": active_workflow,
    }


def render_agent_stats() -> None:
    rows = "".join(
        f'<div class="stat-row"><span>{html.escape(label)}</span>'
        f'<span class="stat-value">{html.escape(str(value))}</span></div>'
        for label, value in _compute_agent_stats().items()
    )
    st.markdown(
        f'<div class="panel-card"><div class="panel-title">Agent Stats</div>{rows}</div>',
        unsafe_allow_html=True,
    )


def render_session_info() -> None:
    rows_data = {
        "Session ID": st.session_state.session_id,
        "Started At": st.session_state.started_at.strftime("%H:%M"),
        "Model": get_model_name(),
        "Temperature": AGENT_TEMPERATURE,
    }
    rows = "".join(
        f'<div class="stat-row"><span>{html.escape(label)}</span>'
        f'<span class="stat-value">{html.escape(str(value))}</span></div>'
        for label, value in rows_data.items()
    )
    st.markdown(
        f'<div class="panel-card"><div class="panel-title">Session Info</div>{rows}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Workflow status / tools-used / file-changes / test-results panel renderers
# (the right-hand dashboard column)
# ---------------------------------------------------------------------------


def render_workflow_status_panel() -> None:
    last_assistant = _last_assistant_message()
    states = last_assistant.get("workflow_states", []) if last_assistant else []
    current_state = states[-1] if states else "IDLE"
    badge_color = {
        "COMPLETED": "green",
        "FAILED": "red",
        "WAITING_FOR_APPROVAL": "orange",
    }.get(current_state, "violet" if states else "gray")

    st.markdown("**🧭 Workflow Status**")
    st.badge(f"Current State: {current_state}", color=badge_color)

    in_progress = (
        states[-1] if states and states[-1] not in ("COMPLETED", "FAILED") else None
    )
    rows = []
    for state, label in WORKFLOW_ROWS:
        if state == in_progress:
            icon, row_class = "🔵", "wf-row active"
        elif state in states:
            icon, row_class = "✅", "wf-row done"
        else:
            icon, row_class = "⚪", "wf-row pending"
        rows.append(f'<div class="{row_class}">{icon} {html.escape(label)}</div>')
    st.markdown(f'<div class="wf-list">{"".join(rows)}</div>', unsafe_allow_html=True)


def render_tools_used_panel() -> None:
    last_assistant = _last_assistant_message()
    tool_calls = last_assistant.get("tool_calls", []) if last_assistant else []
    seen_names: list[str] = []
    for tool_call in tool_calls:
        if tool_call["name"] not in seen_names:
            seen_names.append(tool_call["name"])

    st.markdown("**🧰 Tools Used**")
    if not seen_names:
        st.caption("No tools used in the latest turn yet.")
        return
    chips = "".join(
        f'<div class="tool-chip">✅ {html.escape(TOOL_DISPLAY_NAMES.get(name, name))}</div>'
        for name in seen_names
    )
    st.markdown(f'<div class="tool-chip-list">{chips}</div>', unsafe_allow_html=True)


def render_file_changes_tab() -> None:
    pending = workflow.list_pending_changes()
    applied = _applied_changes_this_session()
    if not pending and not applied:
        st.caption(
            "No file changes yet. Ask the AI to add, fix, or refactor something."
        )
        return

    for change in pending:
        tag = "new" if change.action == "create" else "modified"
        line_count = change.content.count("\n") + 1
        st.markdown(
            f"**{html.escape(change.file_path)}** _({tag})_ · +{line_count} lines · "
            "⏳ pending your approval in the sidebar"
        )
        st.code(change.content, language=_guess_language(change.file_path))

    for change in applied:
        tag = "new" if change.action == "create" else "modified"
        line_count = change.content.count("\n") + 1
        st.markdown(
            f"**{html.escape(change.file_path)}** _({tag})_ · +{line_count} lines · ✅ applied"
        )
        st.code(change.content, language=_guess_language(change.file_path))


def _diff_to_html(diff_text: str) -> str:
    rendered_lines = []
    for line in html.escape(diff_text).splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            rendered_lines.append(f'<span class="diff-add">{line}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            rendered_lines.append(f'<span class="diff-del">{line}</span>')
        else:
            rendered_lines.append(f'<span class="diff-ctx">{line}</span>')
    return '<pre class="diff-block">' + "\n".join(rendered_lines) + "</pre>"


def render_git_diff_tab() -> None:
    if st.button("🔄 Refresh Git Diff", key="refresh_git_diff"):
        try:
            st.session_state.git_diff_cache = tools.git_diff.invoke({})
        except Exception as exc:  # noqa: BLE001 - never crash the UI
            log_error("git_diff_ui", exc)
            st.session_state.git_diff_cache = "⚠️ Could not read the Git diff."

    cached = st.session_state.get("git_diff_cache")
    if cached is None:
        st.caption("Click Refresh to view uncommitted changes in the working tree.")
        return
    st.markdown(_diff_to_html(cached), unsafe_allow_html=True)


def render_project_explorer_tab() -> None:
    if st.button("🔄 Refresh Project Files", key="refresh_project_files"):
        try:
            st.session_state.project_tree_cache = tools.list_project_files.invoke({})
        except Exception as exc:  # noqa: BLE001
            log_error("list_project_files_ui", exc)
            st.session_state.project_tree_cache = (
                "⚠️ Could not read the project structure."
            )

    cached = st.session_state.get("project_tree_cache")
    if cached is None:
        st.caption("Click Refresh to view the project's real file tree.")
        return
    st.code(cached, language=None)


def render_test_results_panel() -> None:
    pytest_calls = _find_tool_calls(st.session_state.messages, "run_pytest")
    st.markdown("**🧪 Test Results**")
    if not pytest_calls:
        st.caption("No tests run yet. Ask the AI to run the test suite.")
        return

    parsed = _parse_pytest_output(pytest_calls[-1]["output"])
    counts = parsed["counts"]
    all_passed = parsed["exit_code"] == 0
    banner_class = "test-banner pass" if all_passed else "test-banner fail"
    banner_text = "✅ ALL TESTS PASSED" if all_passed else "❌ TESTS FAILED"
    duration_text = (
        f"{parsed['duration']}s" if parsed["duration"] is not None else "unknown time"
    )
    st.markdown(
        f'<div class="{banner_class}"><b>{banner_text}</b><br>'
        f"{parsed['total']} total in {duration_text}</div>",
        unsafe_allow_html=True,
    )

    total_col, passed_col, failed_col, skipped_col = st.columns(4)
    total_col.metric("Total", parsed["total"])
    passed_col.metric("Passed", counts["passed"])
    failed_col.metric("Failed", counts["failed"])
    skipped_col.metric("Skipped", counts["skipped"])

    if parsed["tests"]:
        st.caption("Recent tests")
        status_icons = {"PASSED": "✅", "SKIPPED": "⏭️", "FAILED": "❌", "ERROR": "❌"}
        for test in parsed["tests"][:10]:
            icon = status_icons.get(test["status"], "•")
            st.markdown(
                f'<div class="test-row">{icon} {html.escape(test["name"])}</div>',
                unsafe_allow_html=True,
            )
        remaining = len(parsed["tests"]) - 10
        if remaining > 0:
            st.caption(f"...and {remaining} more")


def render_execution_tabs() -> None:
    output_tab, error_tab, log_tab = st.tabs(["🖥 Output", "🩺 Errors", "📜 Logs"])
    pytest_calls = _find_tool_calls(st.session_state.messages, "run_pytest")

    with output_tab:
        if pytest_calls:
            latest_output = str(pytest_calls[-1]["output"])[:4000]
            st.markdown(
                f'<pre class="terminal-block">{html.escape(latest_output)}</pre>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No execution output yet - ask the AI to run the tests.")

    with error_tab:
        repair_attempts = workflow.repair_attempts_snapshot()
        if repair_attempts:
            st.caption("Repair attempts (real-time from the Phase 6 circuit breaker):")
            for file_path, count in repair_attempts.items():
                st.markdown(
                    f"• `{file_path}` — {count} / {workflow.MAX_REPAIR_ATTEMPTS}"
                )

        latest_failure = None
        for tool_call in reversed(pytest_calls):
            if _parse_pytest_output(tool_call["output"])["exit_code"] not in (0, None):
                latest_failure = tool_call
                break
        if latest_failure:
            failed_tests = [
                test
                for test in _parse_pytest_output(latest_failure["output"])["tests"]
                if test["status"] in ("FAILED", "ERROR")
            ]
            if failed_tests:
                st.markdown("**Failed tests:**")
                for test in failed_tests:
                    st.markdown(f"• ❌ `{html.escape(test['name'])}`")
            failure_output = str(latest_failure["output"])[:4000]
            st.markdown(
                f'<pre class="terminal-block error">{html.escape(failure_output)}</pre>',
                unsafe_allow_html=True,
            )
        elif not repair_attempts:
            st.caption("No errors detected in the latest test run.")

    with log_tab:
        events = get_recent_events(limit=200)
        if events:
            lines = [
                f"[{event['time']}] [{event['label']}] {event['text']}"
                for event in events
            ]
            body = html.escape("\n".join(lines))
            st.markdown(
                f'<pre class="terminal-block">{body}</pre>', unsafe_allow_html=True
            )
        else:
            st.caption("No activity logged yet this session.")


def render_final_report_banner() -> None:
    last_assistant = _last_assistant_message()
    if not last_assistant or "COMPLETED" not in last_assistant.get(
        "workflow_states", []
    ):
        return

    tool_calls = last_assistant.get("tool_calls", [])
    created = modified = 0
    for tool_call in tool_calls:
        if tool_call["name"] != "apply_approved_change":
            continue
        change = workflow.get_change(tool_call["input"].get("change_id", ""))
        if change is None:
            continue
        if change.action == "create":
            created += 1
        else:
            modified += 1

    pytest_calls = [tc for tc in tool_calls if tc["name"] == "run_pytest"]
    if pytest_calls:
        parsed = _parse_pytest_output(pytest_calls[-1]["output"])
        tests_text = (
            f"{parsed['counts']['passed']} passed, {parsed['counts']['failed']} failed"
        )
    else:
        tests_text = "—"

    duration = last_assistant.get("duration_seconds")
    duration_text = f"{duration:.0f}s" if duration is not None else "—"

    st.markdown(
        '<div class="final-report-banner">'
        '<div class="final-report-title">🎉 Task Completed Successfully!</div>'
        '<div class="final-report-stats">'
        f"<div><span>Files Changed</span><b>{created + modified} "
        f"({created} new, {modified} modified)</b></div>"
        f"<div><span>Tests</span><b>{html.escape(tests_text)}</b></div>"
        f"<div><span>Time Taken</span><b>{duration_text}</b></div>"
        "<div><span>Status</span><b>COMPLETED</b></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # what gets displayed in the chat

if "lc_history" not in st.session_state:
    st.session_state.lc_history = []  # what gets sent to the agent for context

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "attached_files" not in st.session_state:
    st.session_state.attached_files = []  # documents attached via the chat bar

if "pending_input_source" not in st.session_state:
    st.session_state.pending_input_source = None

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]

if "started_at" not in st.session_state:
    st.session_state.started_at = datetime.now().astimezone()

if "nav_view" not in st.session_state:
    st.session_state.nav_view = "💬 Chat"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🤖 AI Developer Assistant")
    st.caption("Your AI coding companion.")

    render_phase_checklist()

    st.markdown("## 🧭 Navigation")
    st.radio(
        "Navigation",
        ["💬 Chat", "📁 Project Explorer", "🧪 Test Results", "🛠 Tools", "⚙ Settings"],
        key="nav_view",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("## 🎯 Quick Actions")
    mode = st.selectbox(
        "Mode",
        list(MODE_PREFIXES.keys()),
        key="mode_select",
        label_visibility="collapsed",
        help="Choose what kind of help you need, then describe or paste your code below.",
    )

    st.divider()
    st.markdown("## 📎 Attached Files")
    if st.session_state.attached_files:
        for attached in st.session_state.attached_files:
            note = " _(truncated for context)_" if attached.get("truncated") else ""
            st.markdown(
                f"✓ **{attached['filename']}**  \n"
                f"{attached['file_type']} · {documents.format_size(attached['file_size'])}{note}"
            )
        if st.button("🗑 Clear Attachments", use_container_width=True):
            st.session_state.attached_files = []
            st.rerun()
    else:
        st.caption(
            "No files attached. Use the 📎 icon in the chat box below to attach one."
        )

    st.divider()
    st.markdown("## 🔧 Pending Approvals")
    st.caption(
        "Phase 6: file changes the AI has proposed. Nothing is written to "
        "disk until you approve it here."
    )
    pending_changes = workflow.list_pending_changes()
    if pending_changes:
        for change in pending_changes:
            risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                change.risk, "🟡"
            )
            with st.expander(
                f"{risk_icon} {change.action} · {change.file_path} "
                f"(id: {change.change_id})"
            ):
                st.markdown(f"**Reason:** {change.reason}")
                st.markdown(f"**Risk:** {change.risk}")
                st.code(change.content, language="python")
                approve_col, reject_col = st.columns(2)
                with approve_col:
                    if st.button(
                        "✅ Approve",
                        key=f"approve_{change.change_id}",
                        use_container_width=True,
                    ):
                        workflow.approve_change(change.change_id)
                        st.rerun()
                with reject_col:
                    if st.button(
                        "❌ Reject",
                        key=f"reject_{change.change_id}",
                        use_container_width=True,
                    ):
                        workflow.reject_change(change.change_id)
                        st.rerun()
                if change.approved:
                    st.success(
                        "Approved - ask the assistant to apply it "
                        f"(change id `{change.change_id}`)."
                    )
    else:
        st.caption(
            "No pending changes. Ask the AI to add/fix/refactor something to see one here."
        )

    stuck_files = [
        file_path
        for file_path, count in workflow.repair_attempts_snapshot().items()
        if count >= workflow.MAX_REPAIR_ATTEMPTS
    ]
    if stuck_files:
        st.warning(
            "🛑 Repair-attempt limit reached for: "
            + ", ".join(f"`{f}`" for f in stuck_files)
            + ". The AI can no longer apply further automatic fixes to these files "
            "until you reset the counter below."
        )
        if st.button("🔄 Reset repair counter", use_container_width=True):
            workflow.reset_repair_attempts()
            st.rerun()

    st.divider()
    st.markdown("## 🤖 AI Features")
    st.caption(
        "Reference only - browse what each phase can do, then type your own "
        "question in the chat box below. Nothing here sends or runs anything."
    )
    for phase_name, info in AI_FEATURES_INFO.items():
        with st.expander(phase_name):
            st.markdown(info["summary"])
            st.markdown("\n".join(f"- {feature}" for feature in info["features"]))
            st.markdown(f"**Example capability:** _{info['example']}_")

    st.divider()
    render_agent_stats()
    render_session_info()

    st.divider()
    if st.button("🗑 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.session_state.pending_prompt = None
        clear_events()
        st.rerun()

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown(
    '<div style="font-size:1.9rem;font-weight:800;white-space:nowrap;line-height:1.2;">'
    "🤖 AI Developer Assistant Agent</div>"
    '<div style="font-size:0.85rem;color:#9a9ab0;margin-top:2px;">'
    "Advanced Autonomous Software Developer Agent</div>",
    unsafe_allow_html=True,
)
(
    phase_col,
    active_col,
    _spacer_col,
    voice_col,
    upload_col,
    github_col,
    profile_col,
) = st.columns([1, 2, 5, 1, 1, 1, 0.6], vertical_alignment="center")
with phase_col:
    st.badge(CURRENT_PHASE, color="violet")
with active_col:
    st.badge("Agent Status: Active", icon="🟢", color="green")
with voice_col, st.popover("🎙️"):
    st.caption(
        "Use the microphone icon inside the chat box below to record "
        "a voice question."
    )
with upload_col, st.popover("📎"):
    st.caption(
        "Use the 📎 icon inside the chat box below to attach a "
        "document (code, text, PDF, or DOCX)."
    )
with github_col:
    repo_url = _get_repo_url()
    if repo_url:
        st.link_button("🐙", repo_url, help="Open this project's Git remote")
    else:
        st.button("🐙", disabled=True, help="No Git remote configured")
with profile_col:
    st.markdown(
        '<div style="text-align:center;font-size:1.4rem;" '
        'title="Local single-user session">🧑‍💻</div>',
        unsafe_allow_html=True,
    )

st.caption("Ask, generate, debug, review, refactor, and test your code.")
st.divider()

# ---------------------------------------------------------------------------
# API key check (fail loudly but nicely, not with a raw traceback)
# ---------------------------------------------------------------------------

api_key = get_api_key()
if not api_key or api_key == "your_google_api_key_here":
    st.error(
        "⚠️ GOOGLE_API_KEY is not configured. Please add it to your .env file "
        "(see .env.example) and restart the app."
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def get_agent():
    return build_agent()


try:
    agent = get_agent()
except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message below
    log_error("build_agent", exc)
    st.error(
        "⚠️ Could not start the AI agent. Please check your API key and try again."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Main dashboard layout: chat/workflow on the left, live status panels
# (workflow trail, tools used, file changes, test results, execution output)
# on the right - all fed by real session data via the helpers above.
# ---------------------------------------------------------------------------

if st.session_state.nav_view == "💬 Chat":
    chat_col, status_col = st.columns([1.7, 1.4], gap="large")

    with chat_col:
        st.markdown("#### 💬 Chat & Workflow")

        if not st.session_state.messages:
            st.markdown("### 👋 Welcome!")
            st.markdown(
                "I'm your AI Developer Assistant. I can help you with:\n\n"
                "💡 Programming Q&A · 🧮 Calculations · 🐍 Code Explanation\n\n"
                "✍️ Code Generation · 🐞 Debugging · 🔍 Code Review · ♻️ Refactoring\n\n"
                "🧪 Test Generation · ✅ Pytest · 🔎 Ruff · 🎨 Black\n\n"
                "🎤 Voice Questions · 📎 Document Upload (code, text, PDF, DOCX)\n\n"
                "🔧 Multi-step development tasks (Phase 6) - the AI plans, proposes file "
                "changes, and waits for your approval in the sidebar before applying them\n\n"
                "Pick a **Quick Action** in the sidebar, or just ask naturally - "
                'try *"Write a Python function to check whether a number is prime."*'
            )

        # -----------------------------------------------------------------
        # Render chat history
        # -----------------------------------------------------------------

        last_message_index = len(st.session_state.messages) - 1
        for index, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                for tool_call in message.get("tool_calls", []):
                    render_tool_box(tool_call)
                if message["role"] == "assistant":
                    render_answer(message["content"])
                    render_workflow_states(message.get("workflow_states", []))
                    if index == last_message_index:
                        render_execution_plan(message.get("tool_calls", []))
                    render_tts_button(message["content"], key=f"tts_history_{index}")
                else:
                    if message.get("source") == "voice":
                        st.caption("🎤 Voice input · 📝 Transcribed")
                    st.markdown(message["content"])

        # -----------------------------------------------------------------
        # Approval Required: every change still awaiting a decision, straight
        # from the real Phase 6 registry (workflow.list_pending_changes() - the
        # exact same data the sidebar's "Pending Approvals" section already
        # shows). Approve here actually resumes the workflow via
        # _approve_and_resume(); the sidebar's own Approve button is left
        # exactly as-is (mark-approved-only - ask the assistant to apply it).
        # -----------------------------------------------------------------

        chat_pending_changes = workflow.list_pending_changes()
        approved_in_chat = None
        if chat_pending_changes:
            with st.container(border=True):
                st.markdown("**⚠️ Approval Required**")
                st.caption("The following files will be created/modified:")
                for change in chat_pending_changes:
                    tag = "new" if change.action == "create" else "modified"
                    st.markdown(f"• `{change.file_path}` _({tag})_")
                for change in chat_pending_changes:
                    approve_col, reject_col = st.columns(2)
                    with approve_col:
                        if st.button(
                            f"✅ Approve {change.change_id}",
                            key=f"chat_approve_{change.change_id}",
                            use_container_width=True,
                        ):
                            approved_in_chat = change
                    with reject_col:
                        if st.button(
                            f"❌ Reject {change.change_id}",
                            key=f"chat_reject_{change.change_id}",
                            use_container_width=True,
                        ):
                            workflow.reject_change(change.change_id)
                            st.rerun()

        # Rendered outside the narrow approve/reject columns above, so the
        # resulting chat bubbles use the full chat column width like any other
        # turn instead of being squeezed into a half-width column.
        if approved_in_chat is not None:
            _approve_and_resume(agent, approved_in_chat)

        # -----------------------------------------------------------------
        # Handle new input: typed text, an attached document, a recorded voice
        # question, or a prompt clicked from the sidebar - all flow into the
        # exact same agent call below. Voice is only ever converted to text
        # (transcribe_audio) before reaching the agent; attached documents are
        # only ever added as labeled, untrusted reference context - neither
        # path adds a separate AI implementation.
        # -----------------------------------------------------------------

        st.caption(
            "🎤 Ready — use the microphone icon to record a voice question on its "
            "own, or 📎 to attach a document (type a short question alongside it, "
            'e.g. "explain this file", before sending). If your browser denies '
            "microphone access, you can still type your question below."
        )
        chat_value = st.chat_input(
            "Ask your AI Developer Assistant...",
            accept_file="multiple",
            file_type=sorted(documents.SUPPORTED_EXTENSIONS),
            max_upload_size=documents.MAX_UPLOAD_SIZE_MB,
            accept_audio=True,
        )

        user_input = None
        input_source = None  # "voice" when this turn's text came from a transcription

        if chat_value:
            newly_attached = False
            for uploaded in chat_value.files:
                processed = documents.process_upload(uploaded.name, uploaded.getvalue())
                if "error" in processed:
                    st.error(f"❌ {uploaded.name}: {processed['error']}")
                else:
                    # Replace any earlier attachment with the same display name.
                    st.session_state.attached_files = [
                        f
                        for f in st.session_state.attached_files
                        if f["filename"] != processed["filename"]
                    ] + [processed]
                    newly_attached = True

            voice_text = ""
            if chat_value.audio is not None:
                with st.spinner("🔴 Transcribing your voice input..."):
                    try:
                        voice_text = transcribe_audio(
                            chat_value.audio.getvalue(),
                            chat_value.audio.type or "audio/wav",
                        )
                    except Exception as exc:  # noqa: BLE001 - never crash on STT
                        log_error("transcribe_audio", exc)
                        st.error(
                            "🎤 Could not transcribe your voice input. Please try again, "
                            "or continue using the text input."
                        )
                if voice_text:
                    st.info(f'📝 Transcribed: "{voice_text}"')
                elif chat_value.audio is not None:
                    st.warning(
                        "🎤 No speech was detected in that recording. Please try again."
                    )

            text_value = (chat_value.text or "").strip()
            if text_value:
                user_input = text_value
            elif voice_text:
                user_input = voice_text
                input_source = "voice"

            if newly_attached and user_input:
                # The sidebar's "Attached Files" list is rendered earlier in this
                # same script run (top-to-bottom), so it would otherwise still show
                # the pre-upload state for this run. Stash the question and rerun
                # once, the same pending_prompt handoff the sidebar buttons already
                # use, so the sidebar reflects the new attachment before the agent
                # answers.
                st.session_state.pending_prompt = user_input
                st.session_state.pending_input_source = input_source
                st.rerun()

        if st.session_state.pending_prompt and not user_input:
            user_input = st.session_state.pending_prompt
            input_source = st.session_state.pending_input_source
            st.session_state.pending_prompt = None
            st.session_state.pending_input_source = None

        if user_input:
            user_input = user_input.strip()

        if user_input:
            # The chat displays what the user actually typed/said; the agent
            # additionally receives the Quick Action prefix (if any) and any
            # attached-document context, clearly labeled as untrusted reference data.
            prefix = MODE_PREFIXES.get(st.session_state.get("mode_select", "Ask"), "")
            agent_input = f"{prefix}{user_input}" if prefix else user_input
            document_block = documents.build_document_context_block(
                st.session_state.attached_files
            )
            if document_block:
                agent_input = f"{document_block}\n\nUser's question: {agent_input}"

            _run_and_render_turn(agent, user_input, agent_input, source=input_source)

    with status_col:
        with st.container(border=True):
            render_workflow_status_panel()
        with st.container(border=True):
            render_tools_used_panel()

        files_tab, diff_tab, explorer_tab = st.tabs(
            ["📄 Files", "🔀 Diff", "🗂 Explorer"]
        )
        with files_tab:
            render_file_changes_tab()
        with diff_tab:
            render_git_diff_tab()
        with explorer_tab:
            render_project_explorer_tab()

        with st.container(border=True):
            render_test_results_panel()

        render_execution_tabs()
        render_final_report_banner()

elif st.session_state.nav_view == "📁 Project Explorer":
    st.markdown("#### 📁 Project Explorer")
    st.caption("This project's real file tree (read-only) - list_project_files.")
    render_project_explorer_tab()

elif st.session_state.nav_view == "🧪 Test Results":
    st.markdown("#### 🧪 Test Results")
    with st.container(border=True):
        render_test_results_panel()
    render_execution_tabs()

elif st.session_state.nav_view == "🛠 Tools":
    st.markdown("#### 🛠 Tools")
    st.caption(
        "Every tool the agent can call this session. ✅ marks a tool that has "
        "actually run at least once; the rest are available but unused so far."
    )
    used_tool_names = {
        tool_call["name"]
        for message in st.session_state.messages
        for tool_call in message.get("tool_calls", [])
    }
    for tool_name, display_name in sorted(TOOL_DISPLAY_NAMES.items()):
        status = (
            "✅ used this session" if tool_name in used_tool_names else "⚪ available"
        )
        st.markdown(f"{display_name} — _{status}_")

elif st.session_state.nav_view == "⚙ Settings":
    st.markdown("#### ⚙ Settings")
    settings_col1, settings_col2 = st.columns(2)
    with settings_col1:
        render_session_info()
    with settings_col2:
        render_agent_stats()

    st.markdown("**Environment configuration**")
    st.caption(
        "Presence only - actual key/token values are never shown in this UI, "
        "logs, execution output, or Git diffs."
    )
    env_vars = {
        "GOOGLE_API_KEY (required)": "GOOGLE_API_KEY",
        "TAVILY_API_KEY (web search)": "TAVILY_API_KEY",
        "GITHUB_TOKEN (GitHub tools)": "GITHUB_TOKEN",
    }
    for label, var_name in env_vars.items():
        configured = bool(os.getenv(var_name, "").strip())
        icon = "✅" if configured else "⚪"
        state = "configured" if configured else "not configured"
        st.markdown(f"{icon} {label}: _{state}_")
