"""
app.py
------
This file is ONLY responsible for the Streamlit user interface: showing the
chat, taking user input, and displaying the agent's answers. All the AI logic
lives in agent.py, and the tools live in tools.py.

    USER -> STREAMLIT UI -> agent.py (LangChain + Gemini) -> back to UI

UI architecture (v2 - a clean, professional workspace, not a dashboard):
a compact header, a narrow sidebar (9 nav destinations + a Phase 6 footer),
and a single main-content column whose content is chosen by
st.session_state.nav_view. Every value shown anywhere in this file comes
from real session state, workflow.py's real registry, or a real (read-only,
user-triggered) tool call - never invented.
"""

import difflib
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
    describe_agent_error,
    get_api_key,
    get_model_name,
    new_ai_message,
    new_human_message,
    run_agent_turn,
    transcribe_audio,
)
from logger import clear_events, get_recent_events, log_error, log_perf

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
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <!-- No blank lines anywhere inside this <style> block: Streamlit's
         markdown-to-HTML pass treats a raw HTML block as ended by the
         first blank line (CommonMark HTML-block rule), so a blank line in
         here truncates the stylesheet mid-parse and leaks every rule after
         it as literal page text instead of applying it as CSS - a real bug
         found and fixed via visual (Playwright) inspection, not just a
         style preference. Keep every rule on a contiguous, non-blank run
         of lines; use comments (never a blank line) to separate groups. -->
    <style>
    :root {
        --bg: #f7f8fa;
        --surface: #ffffff;
        --border: #e2e5ea;
        --text: #16181d;
        --text-muted: #6b7280;
        --primary: #2563eb;
        --primary-soft: #eaf1ff;
        --success: #16a34a;
        --success-soft: #eafbf1;
        --warning: #b45309;
        --warning-soft: #fdf3e2;
        --error: #dc2626;
        --error-soft: #fdecec;
    }
    html, body, [class*="css"] { font-family: "Inter", -apple-system, sans-serif; }
    code, pre, .mono { font-family: "SFMono-Regular", Consolas, monospace; }
    .stApp { background: var(--bg); }
    .block-container { padding-top: 1.2rem; max-width: 1200px; }
    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    /* Chat responses are rendered from LLM-generated Markdown, which can
       contain ATX headings (#, ##, ...) copied verbatim from scraped
       documentation/web content - a stray "# ..." line otherwise renders
       as a full browser-default heading. These rules clamp every heading
       level to one of two small, bold sizes. */
    [data-testid="stChatMessage"] :is(h1, h2, h3, h4, h5, h6) {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        margin: 0.6rem 0 0.3rem !important;
        line-height: 1.4 !important;
    }
    [data-testid="stChatMessage"] :is(h1, h2) { font-size: 1.15rem !important; }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li { font-size: 1rem !important; line-height: 1.55 !important; }
    [data-testid="stChatMessage"] code { font-size: 0.85em !important; }
    /* Defensive guard: force full brightness regardless of Streamlit's
       internal script-run state, so the page can never be left dimmed
       during/after processing. */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    .block-container, [data-testid="stChatMessage"] {
        opacity: 1 !important; filter: none !important;
    }
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .card-title {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
        color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.6rem;
    }
    .page-title { font-size: 1.35rem; font-weight: 700; margin-bottom: 0.15rem; }
    .page-subtitle { font-size: 0.88rem; color: var(--text-muted); margin-bottom: 1rem; }
    .pill {
        display: inline-flex; align-items: center; gap: 5px;
        font-size: 0.78rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;
        border: 1px solid var(--border);
    }
    .pill.success { color: var(--success); background: var(--success-soft); border-color: transparent; }
    .pill.warning { color: var(--warning); background: var(--warning-soft); border-color: transparent; }
    .pill.error { color: var(--error); background: var(--error-soft); border-color: transparent; }
    .pill.neutral { color: var(--text-muted); background: var(--bg); }
    .task-step { display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 0.86rem; }
    .task-step.done { color: var(--success); }
    .task-step.active { color: var(--primary); font-weight: 600; }
    .task-step.pending { color: var(--text-muted); }
    .row-between { display: flex; justify-content: space-between; align-items: center; }
    .kv-row {
        display: flex; justify-content: space-between; padding: 5px 0;
        font-size: 0.85rem; border-bottom: 1px solid var(--border);
    }
    .kv-row:last-child { border-bottom: none; }
    .kv-row .v { font-weight: 600; }
    .diff-block {
        background: #0d1117; color: #c9d1d9; border: 1px solid var(--border);
        border-radius: 8px; padding: 10px; font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.78rem; white-space: pre-wrap; overflow-x: auto;
        max-height: 420px; overflow-y: auto; margin: 0;
    }
    .diff-add { color: #7ee787; display: block; }
    .diff-del { color: #ffa198; display: block; }
    .diff-ctx { color: #8b949e; display: block; }
    .log-block {
        background: #0d1117; color: #c9d1d9; border: 1px solid var(--border);
        border-radius: 8px; padding: 10px; font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.78rem; white-space: pre-wrap; overflow-x: auto;
        max-height: 460px; overflow-y: auto; margin: 0;
    }
    .log-block.error { color: #ffa198; }
    .shortcut-btn button {
        height: 5.4rem; text-align: left; white-space: pre-line;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

# "Capabilities" panel (sidebar, secondary): pure information about what each
# phase can do - never inserted into the chat input or sent to the agent.
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
    "create_project_zip": "📦 Packaging Project",
    "launch_generated_app": "🚀 Launching Generated Application",
    "stop_generated_app": "🛑 Stopping Generated Application",
}

# "Execution Plan" list: what a turn's tool calls actually did, in order -
# never a forecast, so every line shown already happened.
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
    "create_project_zip": "Package the project into a ZIP",
    "launch_generated_app": "Launch the generated application live",
    "stop_generated_app": "Stop the generated application",
}

# Compact 7-step task strip (Chat page). Each step's "done" state is derived
# only from whether a real WorkflowStatus value in that group actually
# appeared this turn - never a fixed/forecast progression.
_TASK_STEPS = [
    ("Understand requirements", {"PLANNING"}),
    ("Inspect project", {"INSPECTING", "SEARCHING_DOCUMENTATION"}),
    ("Create implementation plan", {"PROPOSING_CHANGE"}),
    ("Waiting for approval", {"WAITING_FOR_APPROVAL"}),
    ("Implement", {"IMPLEMENTING"}),
    ("Test", {"TESTING", "ANALYZING", "FIXING", "RETESTING", "REGRESSION_TESTING"}),
    ("Review", {"REVIEWING", "COMPLETED"}),
    ("Package", {"PACKAGING"}),
    ("Live Preview", {"LIVE_PREVIEW"}),
]

# Home page shortcuts: real prompts sent through the real agent pipeline
# (see _handle_home_shortcut) - never a fake/simulated action.
HOME_SHORTCUTS = [
    (
        "Understand Project",
        "Analyze my codebase",
        (
            "Give me an overview of this project's architecture and how the "
            "main files work together."
        ),
    ),
    (
        "Review Code",
        "Find bugs and issues",
        (
            "Review this project's code for bugs, security issues, and "
            "readability problems."
        ),
    ),
    (
        "Build Feature",
        "Plan and implement",
        (
            "I want to add a new feature to this project. Inspect the project "
            "first, then propose an implementation plan."
        ),
    ),
    (
        "Run Tests",
        "Test and analyze",
        "Run the full test suite and analyze the results.",
    ),
]

# Common tech-stack keywords scanned for in an attached document's real
# extracted text (Documents page). A literal, honest keyword match only -
# never an inferred/guessed requirement for a concept that isn't a keyword.
_REQUIREMENT_KEYWORDS = [
    "FastAPI",
    "Flask",
    "Django",
    "SQLite",
    "PostgreSQL",
    "MySQL",
    "SQLAlchemy",
    "Pydantic",
    "pytest",
    "Ruff",
    "Black",
    "Docker",
    "REST API",
    "JWT",
    "OAuth",
    "authentication",
    "authorization",
    "validation",
    "README",
    "CI/CD",
    "unit test",
    "integration test",
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

NAV_ITEMS = [
    "Home",
    "Chat",
    "Project",
    "Documents",
    "Changes",
    "Tests",
    "Git",
    "Logs",
    "Settings",
]


# ---------------------------------------------------------------------------
# Rendering / formatting helpers
# ---------------------------------------------------------------------------


def _output_block(text: str, limit: int = 2000) -> str:
    """Render tool output as safely-escaped, pre-formatted HTML."""
    text = str(text).strip()
    if len(text) > limit:
        text = text[:limit] + "\n... (truncated)"
    return f'<pre style="white-space:pre-wrap;margin:0;">{html.escape(text)}</pre>'


# Matches a *complete* ```lang\n...\n``` fenced code block. Only fully
# paired fences are matched - a stray/unclosed ``` is left as plain text
# instead of being treated as an (empty) code block.
_CODE_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)


def render_answer(text: str) -> None:
    """Render assistant answer text, drawing fenced code blocks with
    st.code() instead of relying on Markdown's automatic fence detection."""
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
    """Optional 'read aloud' button: the browser's own speechSynthesis
    (Web Speech API) client-side - no server-side TTS service/dependency."""
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
    """A one-line 'Workflow: A -> B' caption for one turn, derived only
    from tools actually called - shown only when Phase 6 was engaged."""
    if not states:
        return
    trail = " → ".join(state.replace("_", " ") for state in states)
    st.caption(f"Workflow: {trail}")


def render_execution_plan(tool_calls: list[dict]) -> None:
    """What this turn's tool calls actually did, in the order they really
    happened - never a forecast, so every line is already checked off."""
    if not tool_calls:
        return
    steps = [
        EXECUTION_STEP_LABELS.get(tool_call["name"], tool_call["name"])
        for tool_call in tool_calls
    ]
    with st.expander("Steps taken this turn", expanded=False):
        for index, step in enumerate(steps, start=1):
            st.markdown(f"{index}. {html.escape(step)} ✓")


def render_tool_box(tool_call: dict) -> None:
    """Render a single 'Using X' note for one tool call."""
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
        f'<div class="card" style="border-left:3px solid var(--primary);">'
        f"<b>{header}</b><br>{body}</div>",
        unsafe_allow_html=True,
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
    `agent_input`, e.g. an approval nudge or a Home-page shortcut prompt) -
    shared by every real turn trigger in this file (typed input, a Home
    shortcut, or an Approve click) so none of them duplicate this logic.
    """
    st.session_state.messages.append(
        {"role": "user", "content": displayed_text, "tool_calls": [], "source": source}
    )
    st.session_state.lc_history.append(new_human_message(agent_input))

    with st.chat_message("user"):
        if source == "voice":
            st.caption("🎤 Voice input · transcribed")
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
                answer = f"⚠️ {describe_agent_error(exc)}"
                tool_calls = []
                workflow_states = []
        turn_duration = time.time() - turn_started_at

        for tool_call in tool_calls:
            render_tool_box(tool_call)
            if tool_call.get("name") == "apply_approved_change":
                _note_applied_change(
                    tool_call.get("input", {}).get("change_id", "")
                )

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


def _approve_changeset(changeset_id: str) -> None:
    """Approve every member of ONE Phase 6 change set as a single human
    action (workflow.approve_change_set) - the entire simplified UI's first
    of exactly two actions. Deliberately does NOT modify files, run tests,
    or call apply_approved_change: Approve only moves state from PENDING to
    APPROVED, nothing else - Apply (see _apply_changeset_and_resume) is a
    separate, later action.

    workflow.approve_change_set() is idempotent: it returns the change_ids
    it actually just approved, empty if the whole set was already approved
    (e.g. a duplicate click, or a Streamlit rerun re-delivering the same
    click event) - in that case this is a deliberate no-op, so a single
    click can never announce, or count, the same approval twice.
    """
    newly_approved = workflow.approve_change_set(changeset_id)
    if not newly_approved:
        return
    if len(newly_approved) == 1:
        message = "✅ The proposed change has been approved."
    else:
        message = f"✅ All {len(newly_approved)} proposed changes have been approved."
    st.session_state.messages.append(
        {"role": "assistant", "content": message, "tool_calls": []}
    )
    st.rerun()


def _apply_changeset_and_resume(agent, changeset_id: str) -> None:
    """Deterministically write EVERY approved-but-not-yet-applied member of
    one change set to disk exactly once (tools.apply_approved_change_set -
    never left to the AI agent's own tool-calling judgment to remember to
    call apply_approved_change once per file), then drive the SAME agent
    through run_agent_turn's existing bounded auto-continuation loop to run
    the tests, analyze/fix, and report - exactly what Phase 6 steps 5+
    already do, triggered automatically instead of requiring a follow-up
    message.

    Idempotent: if every member is already applied (e.g. a duplicate click,
    or a Streamlit rerun re-delivering the same click), apply_approved_
    change_set() reports nothing newly applied and this is a deliberate
    no-op - no second "applied successfully" message, no second agent turn,
    no file written twice.
    """
    outcome = tools.apply_approved_change_set(changeset_id)
    applied = outcome["applied"]
    failed = outcome["failed"]
    if not applied and not failed:
        return  # already fully applied - nothing left to do

    if not applied:
        # Nothing was actually written (e.g. the repair-attempt limit was
        # already reached) - say so plainly. No agent turn to nudge into
        # testing changes that were never written.
        detail = "; ".join(f"{e['file_path']}: {e['message']}" for e in failed)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"⚠️ Could not apply the approved changes. {detail}",
                "tool_calls": [],
            }
        )
        st.rerun()
        return

    # Record every change this batch actually wrote to disk, in the real
    # order apply_approved_change_set applied them - the same real-time log
    # _run_and_render_turn appends to for the agent's own apply_approved_change
    # tool calls, so _current_generated_project_root()/_applied_changes_
    # this_session() see this application correctly no matter which of the
    # two ways a file actually got written to disk.
    for entry in applied:
        _note_applied_change(entry["change_id"])

    summary = (
        f"✅ {len(applied)} approved change{'s' if len(applied) != 1 else ''} "
        "applied successfully."
    )
    if failed:
        summary += " ⚠️ " + "; ".join(
            f"{e['file_path']}: {e['message']}" for e in failed
        )

    change_list = ", ".join(f"{e['change_id']} ({e['file_path']})" for e in applied)
    nudge = (
        f"The following approved change(s) have already been applied to disk: "
        f"{change_list}. Run the complete test suite now, analyze any failure, "
        "propose a fix if needed (it will need its own approval), and continue "
        "the existing Phase 6 workflow (regression testing, Ruff, Black, final "
        "verification). Finish with a concise summary."
    )
    _run_and_render_turn(agent, summary, nudge, spinner_text="Applying and testing...")
    st.rerun()


# ---------------------------------------------------------------------------
# Data helpers - every panel in this file reads ONLY real session data:
# messages already produced this session (each carrying the tool_calls/
# workflow_states that turn actually produced) and workflow.py's real
# pending/applied-change registry. Nothing here invents a number.
# ---------------------------------------------------------------------------


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
    id even after it's applied - only list_pending_changes() filters those
    out, since that list is for the *pending* approval queue). Sourced from
    st.session_state.applied_change_ids - see _current_generated_project_root
    for why that log, not chat-message scanning, is the correct source."""
    seen_ids: set[str] = set()
    applied: list[workflow.ProposedChange] = []
    for change_id in st.session_state.applied_change_ids:
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


def _unified_diff(old_text: str, new_text: str, file_path: str) -> str:
    """A real diff (Python's stdlib difflib) between the actual on-disk
    content and a proposed change's content - computed here, in the UI,
    from real text on both sides; never a fabricated/simulated diff."""
    diff_lines = difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )
    return "\n".join(diff_lines)


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


_PYTEST_TEST_LINE_RE = re.compile(
    r"^(\S+::\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)\b", re.MULTILINE
)
_PYTEST_SUMMARY_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|skipped|errors?)\b")
_PYTEST_DURATION_RE = re.compile(r"\bin\s+([\d.]+)s\b")
_PYTEST_EXIT_CODE_RE = re.compile(r"^Exit code:\s*(\d+)")


def _parse_pytest_output(output: str) -> dict:
    """Pull real counts/durations/per-test results out of run_pytest's own
    output. Never guesses a number it can't find - missing fields stay
    0/None."""
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


def _current_task() -> dict | None:
    """The active development task, if one exists this session: the human
    text that most recently produced real workflow_states, plus the real
    states from that turn's assistant reply. None if no Phase 6 workflow
    has been engaged yet - never a placeholder task."""
    messages = st.session_state.messages
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message["role"] != "assistant" or not message.get("workflow_states"):
            continue
        task_text = None
        for earlier in reversed(messages[:index]):
            if earlier["role"] == "user":
                task_text = earlier["content"]
                break
        return {
            "task": task_text or "Development task",
            "states": message["workflow_states"],
        }
    return None


def _task_progress(states: list[str]) -> list[tuple[str, str]]:
    """Map real workflow_states onto the compact task strip.
    Returns (label, status) where status is 'done'/'active'/'pending'."""
    state_set = set(states)
    current = states[-1] if states else None
    rows = []
    for label, group in _TASK_STEPS:
        if current in group:
            status = "active"
        elif state_set & group:
            status = "done"
        else:
            status = "pending"
        rows.append((label, status))
    return rows


def _render_project_package_section(task: dict | None) -> None:
    """Once this session's development task has actually reached the real
    COMPLETED workflow state - see agent.py's run_agent_turn, which derives
    COMPLETED only from a genuine passing run_pytest with nothing else
    pending - offer a verified, downloadable ZIP of the project's current
    real files. Never shown before that point, and never a fabricated
    "success" - if the project can't actually be packaged, this shows a
    real error instead of a download button.

    If this task actually built a generated sub-project (resolved the same
    way _render_live_application_section resolves it, via
    _current_generated_project_root/_generated_project_tests_passed), the
    archive packages ONLY that sub-project's own files
    (tools.get_or_build_generated_project_zip, source_dir=that project's
    real folder) - e.g. "todo_app.zip" containing just
    generated_projects/todo_app/, never this whole assistant's own
    repository. Only a task that changed THIS assistant's own project falls
    back to tools.get_or_build_project_zip() (the whole-project archive).

    Both are called directly (never through the agent), the same way the
    Project/Git pages already call read-only tools directly for a plain
    button click - and both are themselves idempotent (see their
    docstrings), so re-rendering this section on every Streamlit rerun never
    rebuilds/re-saves the archive unless the relevant project's real files
    actually changed.
    """
    if not task or "COMPLETED" not in task["states"]:
        return

    generated_project_root = _current_generated_project_root()
    if generated_project_root:
        if not _generated_project_tests_passed(generated_project_root):
            return  # never offer a package unless that project's own tests passed
        tests_passed = True
    else:
        pytest_calls = _find_tool_calls(st.session_state.messages, "run_pytest")
        if not pytest_calls:
            return
        tests_passed = (
            _parse_pytest_output(pytest_calls[-1]["output"])["exit_code"] == 0
        )
        if not tests_passed:
            return  # never offer a package unless the real test suite actually passed

    ruff_calls = _find_tool_calls(st.session_state.messages, "run_ruff")
    black_calls = _find_tool_calls(st.session_state.messages, "run_black")
    ruff_passed = (
        "no issues" in str(ruff_calls[-1]["output"]).lower() if ruff_calls else None
    )
    black_passed = (
        not str(black_calls[-1]["output"]).lower().startswith("error")
        if black_calls
        else None
    )

    def _status_pill(label: str, state: bool | None) -> str:
        if state is None:
            return f'<span class="pill neutral">{label}: Not run</span>'
        pill_class = "success" if state else "error"
        return f'<span class="pill {pill_class}">{label}: {"PASS" if state else "FAIL"}</span>'

    st.write("")
    with st.container(border=True):
        st.markdown("#### 📦 Project Completed ✅")
        st.markdown(
            " &nbsp; ".join(
                [
                    _status_pill("Tests", tests_passed),
                    _status_pill("Ruff", ruff_passed),
                    _status_pill("Black", black_passed),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.write("")

        try:
            if generated_project_root:
                slug = generated_project_root.rsplit("/", 1)[-1]
                package = tools.get_or_build_generated_project_zip(
                    source_dir=generated_project_root, project_name=slug
                )
            else:
                package = tools.get_or_build_project_zip(project_name=task["task"])
        except OSError as exc:
            log_error("get_or_build_project_zip_ui", exc)
            st.error("⚠️ Could not create the project archive.")
            return

        secrets_note = (
            f"{package['excluded_secrets']} file(s) excluded"
            if package["excluded_secrets"]
            else "None found"
        )
        st.markdown("**PROJECT PACKAGE**")
        st.markdown("✅ ZIP Ready")
        st.caption(
            f"File: {package['filename']} · "
            f"Size: {documents.format_size(len(package['bytes']))}"
        )
        st.caption(
            f"Files included: {package['included']} · "
            f"Files excluded: {package['excluded']} · "
            f"Secrets excluded: {secrets_note}"
        )
        st.download_button(
            "📦 Download Project ZIP",
            data=package["bytes"],
            file_name=package["filename"],
            mime="application/zip",
            use_container_width=True,
            key="download_project_zip",
        )


def _note_applied_change(change_id: str) -> None:
    """Record one change_id as applied to disk - the single source
    st.session_state.applied_change_ids has always been populated from
    (whether the write happened via the agent's own apply_approved_change
    tool call or via this UI's batch "Apply Approved Changes" button; see
    _run_and_render_turn/_apply_changeset_and_resume) - and, if that change
    just wrote a file under "generated_projects/<slug>", make that project
    the explicitly-tracked active one (st.session_state.active_project_path).

    This is the ONLY place active_project_path is written, and it fires at
    exactly the moment a generated project's file is actually applied -
    i.e. whenever a new project is generated (its first applied file makes
    it active) and whenever generation of further files completes. The
    Project tab (render_project_page) reads active_project_path alone -
    never by scanning every folder under generated_projects/."""
    st.session_state.applied_change_ids.append(change_id)
    change = workflow.get_change(change_id)
    if change is None or not change.applied:
        return
    file_path = change.file_path
    if not file_path.startswith("generated_projects/"):
        return
    parts = file_path.split("/")
    if len(parts) >= 2:
        st.session_state.active_project_path = "/".join(parts[:2])


def _current_generated_project_root() -> str | None:
    """The generated_projects/<slug> root this session currently treats as
    active, for the packaging/live-preview sections (_render_project_
    package_section/_render_live_application_section) - the same explicitly
    tracked st.session_state.active_project_path the Project tab uses (see
    _note_applied_change), never guessed from chat text or re-derived by
    scanning every folder under generated_projects/."""
    return st.session_state.active_project_path


def _generated_project_tests_passed(project_root: str) -> bool:
    """True only if the most recent run_pytest call actually targeting this
    exact generated project's own tests reported a real passing exit code -
    never inferred from anything else."""
    pytest_calls = [
        tool_call
        for tool_call in _find_tool_calls(st.session_state.messages, "run_pytest")
        if str(tool_call["input"].get("target", "")).startswith(project_root)
    ]
    if not pytest_calls:
        return False
    return _parse_pytest_output(pytest_calls[-1]["output"])["exit_code"] == 0


def _render_live_application_section() -> None:
    """Once a generated Streamlit project's own tests have genuinely
    passed, offer to launch/stop it as its own separate local server - via
    tools.launch_generated_app/stop_generated_app called directly (never
    through the agent), the same way the Changes page's Approve/Reject and
    repair-counter-reset buttons already call workflow.py functions
    directly. Never shown before that point, and the URL shown is always
    read back from tools.get_generated_server's own tracked state - never
    built from model output."""
    project_root = _current_generated_project_root()
    if not project_root or not _generated_project_tests_passed(project_root):
        return

    slug = project_root.rsplit("/", 1)[-1]
    server = tools.get_generated_server(project_root)

    st.write("")
    with st.container(border=True):
        st.markdown("#### 🚀 Live Application")
        st.caption(f"Project: {html.escape(slug)}")

        if server is None or server.status == "stopped":
            st.markdown(
                '<span class="pill neutral">Not running</span>',
                unsafe_allow_html=True,
            )
            if st.button("Start Live App", key=f"start_live_{slug}"):
                with st.spinner("Starting the generated application..."):
                    tools.launch_generated_app.invoke({"project_root": project_root})
                st.rerun()
        elif server.status == "running":
            st.markdown(
                '<span class="pill success">🟢 Running</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"URL: {server.url}")
            open_col, stop_col = st.columns(2)
            with open_col:
                st.link_button("Open Live App", server.url, use_container_width=True)
            with stop_col:
                if st.button(
                    "Stop App", key=f"stop_live_{slug}", use_container_width=True
                ):
                    tools.stop_generated_app.invoke({"project_root": project_root})
                    st.rerun()
        else:  # "failed"
            st.markdown(
                '<span class="pill error">Failed to start</span>',
                unsafe_allow_html=True,
            )
            st.error(server.error or "The generated application could not be started.")
            if st.button("Retry", key=f"retry_live_{slug}"):
                with st.spinner("Starting the generated application..."):
                    tools.launch_generated_app.invoke({"project_root": project_root})
                st.rerun()


def _request_nav_change(page: str) -> None:
    """Switch pages from a button click anywhere outside the sidebar's own
    radio widget. Streamlit forbids writing st.session_state.nav_view
    directly once that widget has rendered in this run, so this stashes the
    request in a plain (non-widget) key that the sidebar consumes on the
    NEXT run, right before it re-creates the radio - the same handoff
    pattern already used for pending_prompt."""
    st.session_state._nav_request = page
    st.rerun()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # what gets displayed in the chat

if "applied_change_ids" not in st.session_state:
    # Every change_id actually written to disk this session, in the true
    # order it was applied - recorded directly at the moment of application
    # (see _run_and_render_turn and _apply_changeset_and_resume), regardless
    # of whether the write happened via the agent's own apply_approved_change
    # tool call or via this UI's batch "Apply Approved Changes" button. This
    # is the one source _current_generated_project_root() and
    # _applied_changes_this_session() both read from - never reconstructed
    # by scanning chat text.
    st.session_state.applied_change_ids = []

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
    st.session_state.nav_view = "Chat"  # Chat is the primary page

if "_nav_request" not in st.session_state:
    st.session_state._nav_request = None

if "git_diff_cache" not in st.session_state:
    st.session_state.git_diff_cache = None

if "git_status_cache" not in st.session_state:
    st.session_state.git_status_cache = None

if "git_branch_cache" not in st.session_state:
    st.session_state.git_branch_cache = None

if "active_project_path" not in st.session_state:
    # The one, explicitly tracked PROJECT_ROOT-relative path (e.g.
    # "generated_projects/todo_list") the Project tab shows - set only by
    # _note_applied_change, the moment a generated project's file is
    # actually applied to disk. None means no project has been generated
    # yet this session.
    st.session_state.active_project_path = None

if "project_tree_cache" not in st.session_state:
    st.session_state.project_tree_cache = None

if "project_tree_cache_root" not in st.session_state:
    # Which active_project_path project_tree_cache was actually built for -
    # lets the Project tab notice the active project changed (e.g. a new
    # project was just generated) and rebuild automatically, without
    # requiring a manual Refresh click.
    st.session_state.project_tree_cache_root = None


# ---------------------------------------------------------------------------
# Sidebar - narrow, clean: title, 9 nav destinations, Phase 6 footer.
# Phase 1-6 detail lives in a secondary, collapsed "Capabilities" expander,
# never as the primary navigation.
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div style="font-weight:700;font-size:1.05rem;">AI Developer Assistant</div>',
        unsafe_allow_html=True,
    )
    st.caption("Your AI coding companion")
    st.write("")

    # A page switch requested by a button click elsewhere (Review changes,
    # the header settings icon, a Home shortcut) is handed off via
    # _nav_request and consumed HERE, before the radio widget below is
    # instantiated - Streamlit forbids writing to a widget-bound
    # session_state key (nav_view) after that widget has already rendered
    # in the same run, so those click handlers never write nav_view
    # directly (see _request_nav_change()).
    if st.session_state._nav_request is not None:
        st.session_state.nav_view = st.session_state._nav_request
        st.session_state._nav_request = None

    st.radio("Navigate", NAV_ITEMS, key="nav_view", label_visibility="collapsed")

    st.write("")
    with st.expander("Capabilities"):
        st.caption(
            "Reference only - browse what each phase can do, then use the "
            "app normally. Nothing here sends or runs anything."
        )
        for phase_name, info in AI_FEATURES_INFO.items():
            with st.expander(phase_name):
                st.markdown(info["summary"])
                st.markdown("\n".join(f"- {feature}" for feature in info["features"]))
                st.markdown(f"**Example capability:** _{info['example']}_")

    if st.session_state.attached_files:
        st.write("")
        st.caption(f"{len(st.session_state.attached_files)} document(s) attached")

    pending_count = len(workflow.list_pending_changes())
    if pending_count:
        st.write("")
        st.warning(f"{pending_count} change(s) awaiting your review")
        if st.button("Review changes", use_container_width=True):
            _request_nav_change("Changes")

    st.markdown(
        '<div style="position:sticky;bottom:0;padding-top:1rem;">'
        '<div style="font-size:0.72rem;color:var(--text-muted);'
        'text-transform:uppercase;letter-spacing:0.03em;">Phase 6</div>'
        '<div style="font-size:0.85rem;font-weight:600;">Autonomous Developer Agent</div>'
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header - compact, single row.
# ---------------------------------------------------------------------------

header_left, header_right = st.columns([5, 1], vertical_alignment="center")
with header_left:
    st.markdown(
        '<div style="font-size:1.4rem;font-weight:700;">AI Developer Assistant</div>'
        '<div style="font-size:0.85rem;color:var(--text-muted);">'
        "Your AI-powered software development assistant</div>",
        unsafe_allow_html=True,
    )
with header_right:
    status_col, settings_col = st.columns([2, 1], vertical_alignment="center")
    with status_col:
        st.markdown(
            '<div class="pill success" style="justify-content:center;">● Online</div>',
            unsafe_allow_html=True,
        )
    with settings_col:
        if st.button("⚙", key="header_settings", help="Settings"):
            _request_nav_change("Settings")

st.divider()

# ---------------------------------------------------------------------------
# API key check (fail loudly but nicely, not with a raw traceback)
# ---------------------------------------------------------------------------

api_key = get_api_key()
if not api_key or not api_key.strip() or api_key == "your_google_api_key_here":
    st.error(
        "⚠️ GOOGLE_API_KEY is not configured. Please add it to your .env file "
        "(see .env.example) and restart the app."
    )
    st.stop()
# Deliberately no UI warning for a key that merely doesn't match the
# traditional "AIza..." shape (api_key_looks_valid is advisory only - see
# its own docstring) - other real Google credential shapes are known to
# work, and only an actual rejection from Google's API (surfaced below via
# build_agent()'s own exception handling) should ever tell the user
# something is wrong with their key.


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
# Home page
# ---------------------------------------------------------------------------


def _handle_home_shortcut(prompt_text: str) -> None:
    """A Home shortcut sends a real prompt through the same pipeline as
    typing it in Chat - reuses the existing pending_prompt handoff so the
    Chat page picks it up and answers on the very next run."""
    st.session_state.pending_prompt = prompt_text
    st.session_state.pending_input_source = None
    _request_nav_change("Chat")


def render_home_page() -> None:
    st.markdown(
        '<div class="page-title">AI Developer Assistant</div>'
        '<div class="page-subtitle">Build, understand, test and improve your '
        "software with AI.</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for col, (title, subtitle, prompt_text) in zip(cols, HOME_SHORTCUTS):
        with col:
            st.markdown('<div class="shortcut-btn">', unsafe_allow_html=True)
            if st.button(
                f"{title}\n{subtitle}",
                key=f"home_{title}",
                use_container_width=True,
            ):
                _handle_home_shortcut(prompt_text)
            st.markdown("</div>", unsafe_allow_html=True)

    task = _current_task()
    if task:
        st.write("")
        st.markdown(
            '<div class="card-title">Current task</div>', unsafe_allow_html=True
        )
        with st.container(border=True):
            st.markdown(f"**{html.escape(task['task'])}**")
            for label, status in _task_progress(task["states"]):
                icon = {"done": "✓", "active": "●", "pending": "○"}[status]
                st.markdown(
                    f'<div class="task-step {status}">{icon} {html.escape(label)}</div>',
                    unsafe_allow_html=True,
                )
        _render_project_package_section(task)
        _render_live_application_section()


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------


def render_chat_page() -> None:
    st.markdown(
        '<div class="row-between"><div class="page-title">Developer Assistant</div>'
        '<div class="pill success">● Ready</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if not st.session_state.messages:
        st.markdown(
            "Ask me to explain code, generate a function, review your project, "
            "run your tests, or build a whole feature end to end. Try:\n\n"
            '*"Write a Python function to check whether a number is prime."*'
        )

    task = _current_task()
    if task:
        with st.container(border=True):
            st.markdown(f"**Task**  \n{html.escape(task['task'])}")
            st.write("")
            progress_cols = st.columns(len(_task_progress(task["states"])))
            for col, (label, status) in zip(
                progress_cols, _task_progress(task["states"])
            ):
                icon = {"done": "✓", "active": "●", "pending": "○"}[status]
                with col:
                    st.markdown(
                        f'<div class="task-step {status}" style="justify-content:center;">'
                        f"{icon} {html.escape(label)}</div>",
                        unsafe_allow_html=True,
                    )
        _render_project_package_section(task)
        _render_live_application_section()
        st.write("")

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
                    st.caption("🎤 Voice input · transcribed")
                st.markdown(message["content"])

    _render_pending_changeset(agent)

    chat_value = st.chat_input(
        "Ask your developer assistant...",
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
            doc_started_at = time.time()
            processed = documents.process_upload(uploaded.name, uploaded.getvalue())
            log_perf(
                f"Document processing ({uploaded.name})", time.time() - doc_started_at
            )
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
            # Stash the question and rerun once so the sidebar's attachment
            # count (rendered earlier in this same run) reflects the new
            # attachment before the agent answers.
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
        # additionally receives any attached-document context, clearly
        # labeled as untrusted reference data.
        agent_input = user_input
        document_block = documents.build_document_context_block(
            st.session_state.attached_files
        )
        if document_block:
            agent_input = f"{document_block}\n\nUser's question: {agent_input}"

        _run_and_render_turn(agent, user_input, agent_input, source=input_source)


# ---------------------------------------------------------------------------
# Project page
# ---------------------------------------------------------------------------


def _load_active_project_tree(project_root: str) -> str:
    """The active generated project's real tree (tools.get_generated_project_
    tree, called directly - never through the agent), or a plain error
    string if it can't be read. Never raises."""
    try:
        return tools.get_generated_project_tree(project_root)
    except Exception as exc:  # noqa: BLE001
        log_error("get_generated_project_tree_ui", exc)
        return "⚠️ Could not read the project structure."


def render_project_page() -> None:
    st.markdown('<div class="page-title">Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">The active generated project\'s real '
        "file tree (read-only).</div>",
        unsafe_allow_html=True,
    )

    refresh_clicked = st.button("Refresh", key="refresh_project_files")

    # Renders ONLY st.session_state.active_project_path - the one,
    # explicitly tracked current/active project (see _note_applied_change) -
    # never every folder under generated_projects/.
    active_project = st.session_state.active_project_path

    if not active_project:
        st.session_state.project_tree_cache = None
        st.session_state.project_tree_cache_root = None
        st.caption(
            "No project selected. Generate or select a project to view its "
            "structure."
        )
        return

    # Rebuild whenever asked (Refresh) or whenever the active project has
    # changed since the cached tree was built (e.g. a new project was just
    # generated) - so switching projects shows the new one immediately,
    # with no manual Refresh required.
    if refresh_clicked or st.session_state.project_tree_cache_root != active_project:
        st.session_state.project_tree_cache = _load_active_project_tree(active_project)
        st.session_state.project_tree_cache_root = active_project

    st.code(st.session_state.project_tree_cache, language=None)


# ---------------------------------------------------------------------------
# Documents page
# ---------------------------------------------------------------------------


def render_documents_page() -> None:
    st.markdown('<div class="page-title">Documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Documents attached via the Chat page\'s '
        "attach control.</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.attached_files:
        st.caption(
            "No documents attached. Attach one from the + control in the Chat "
            "page's input box."
        )
        return

    for attached in st.session_state.attached_files:
        with st.container(border=True):
            st.markdown(f"**{html.escape(attached['filename'])}**")
            st.caption(
                f"{attached['file_type']} · "
                f"{documents.format_size(attached['file_size'])}"
                + (" · truncated for context" if attached.get("truncated") else "")
            )
            st.markdown(
                '<span class="pill success">✓ Read</span> '
                '<span class="pill success">✓ Text extracted</span>',
                unsafe_allow_html=True,
            )

            view_tab, req_tab = st.tabs(["View document", "View requirements"])
            with view_tab:
                st.text_area(
                    "Extracted text",
                    attached["content"],
                    height=220,
                    key=f"doc_view_{attached['filename']}",
                    label_visibility="collapsed",
                )
            with req_tab:
                content_lower = attached["content"].lower()
                hits = [
                    keyword
                    for keyword in _REQUIREMENT_KEYWORDS
                    if keyword.lower() in content_lower
                ]
                if hits:
                    st.caption(
                        "Keyword matches found in the actual extracted text "
                        "(not a full requirements analysis):"
                    )
                    for keyword in hits:
                        st.markdown(f"✓ {keyword}")
                else:
                    st.caption(
                        "No recognized tech-stack keywords found in the "
                        "extracted text."
                    )

    if st.button("Clear all attachments"):
        st.session_state.attached_files = []
        st.rerun()


# ---------------------------------------------------------------------------
# Changes page - the one, consolidated Proposed Changes view.
# ---------------------------------------------------------------------------


def _render_change_detail(change: workflow.ProposedChange) -> None:
    """Read-only detail block for one proposed change within a change set -
    file path, action, risk level, purpose, change ID, and a diff/content
    preview. No Approve/Reject control of its own: approval and apply are
    always whole-change-set actions (see _render_pending_changeset) - never
    rendered per file, so a task proposing several files never shows more
    than one Approve button and one Apply button in total."""
    risk_pill_class = {"low": "success", "medium": "warning", "high": "error"}.get(
        change.risk, "warning"
    )
    st.markdown(
        f"**{html.escape(change.file_path)}** &nbsp; "
        f'<span class="pill neutral">{change.action.upper()}</span> &nbsp;'
        f'<span class="pill {risk_pill_class}">Risk: {change.risk.title()}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Purpose: {change.reason}")
    st.caption(f"Change ID: `{change.change_id}`")

    with st.expander("Review changes"):
        if change.action == "modify":
            safe_path = tools.PROJECT_ROOT / change.file_path
            try:
                old_text = safe_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                old_text = ""
            diff_text = _unified_diff(old_text, change.content, change.file_path)
            if diff_text.strip():
                st.markdown(_diff_to_html(diff_text), unsafe_allow_html=True)
            else:
                st.caption("No textual difference detected.")
        else:
            st.caption("New file - full proposed content:")
            st.code(change.content, language=_guess_language(change.file_path))


def _render_pending_changeset(agent) -> None:
    """Render the ONE current Phase 6 change set, if any, as a single
    reviewable card - full detail for every proposed file, but exactly ONE
    "Approve Changes" button for the whole set (never one button per file),
    and, only once every member is approved, exactly ONE "Apply Approved
    Changes" button. Shared by the Chat page (inline, right where the
    proposal appeared) and the dedicated Changes page, so a proposal is
    never reviewable in only one of the two places.

    A change set stops being "current" (workflow.get_current_changeset_id()
    returns None) the instant every one of its members has been applied -
    at that point this renders nothing at all, so there is never a
    lingering Approve/Apply control that could repeat an already-finished
    action. Reject discards the whole set at once, the same way Approve
    approves it - never per file.
    """
    changeset_id = workflow.get_current_changeset_id()
    if changeset_id is None:
        return

    pending_view = [c for c in workflow.list_changeset(changeset_id) if not c.applied]
    if not pending_view:
        return
    all_approved = all(c.approved for c in pending_view)

    with st.container(border=True):
        st.markdown(
            '<div class="card-title">Proposed Changes</div>', unsafe_allow_html=True
        )
        for change in pending_view:
            _render_change_detail(change)
        st.caption(f"TOTAL: {len(pending_view)} change(s)")

        if not all_approved:
            approve_col, reject_col = st.columns(2)
            with approve_col:
                if st.button(
                    "✅ Approve Changes",
                    key=f"approve_set_{changeset_id}",
                    use_container_width=True,
                    type="primary",
                ):
                    _approve_changeset(changeset_id)
            with reject_col:
                if st.button(
                    "Reject Changes",
                    key=f"reject_set_{changeset_id}",
                    use_container_width=True,
                ):
                    workflow.reject_change_set(changeset_id)
                    st.rerun()
        else:
            st.success(f"Changes approved ✅ — {len(pending_view)} change(s) approved.")
            if st.button(
                "🚀 Apply Approved Changes",
                key=f"apply_set_{changeset_id}",
                use_container_width=True,
                type="primary",
            ):
                _apply_changeset_and_resume(agent, changeset_id)


def render_changes_page() -> None:
    st.markdown(
        '<div class="page-title">Proposed Changes</div>', unsafe_allow_html=True
    )

    pending = workflow.list_pending_changes()
    if not pending:
        st.markdown(
            '<div class="page-subtitle">No pending changes. Ask the assistant to '
            "add, fix, or refactor something to see one here.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="page-subtitle">{len(pending)} file(s) pending your '
            "review.</div>",
            unsafe_allow_html=True,
        )

    _render_pending_changeset(agent)

    stuck_files = [
        file_path
        for file_path, count in workflow.repair_attempts_snapshot().items()
        if count >= workflow.MAX_REPAIR_ATTEMPTS
    ]
    if stuck_files:
        st.warning(
            "Repair-attempt limit reached for: "
            + ", ".join(f"`{f}`" for f in stuck_files)
            + ". The AI can no longer apply further automatic fixes to these files "
            "until you reset the counter below."
        )
        if st.button("Reset repair counter"):
            workflow.reset_repair_attempts()
            st.rerun()

    applied = _applied_changes_this_session()
    if applied:
        st.write("")
        st.markdown(
            '<div class="card-title">Applied this session</div>', unsafe_allow_html=True
        )
        for change in applied:
            st.markdown(
                f"✓ **{html.escape(change.file_path)}** "
                f'<span class="pill neutral">{change.action.upper()}</span>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tests page
# ---------------------------------------------------------------------------


def render_tests_page() -> None:
    st.markdown('<div class="page-title">Tests</div>', unsafe_allow_html=True)

    pytest_calls = _find_tool_calls(st.session_state.messages, "run_pytest")
    if not pytest_calls:
        st.markdown(
            '<div class="page-subtitle">No tests run yet. Ask the assistant to run '
            "the test suite.</div>",
            unsafe_allow_html=True,
        )
        return

    parsed = _parse_pytest_output(pytest_calls[-1]["output"])
    counts = parsed["counts"]
    all_passed = parsed["exit_code"] == 0
    pill_class = "success" if all_passed else "error"
    summary = f"✓ {counts['passed']} passed" + (
        "" if all_passed else f" · ✗ {counts['failed']} failed"
    )
    st.markdown(
        f'<span class="pill {pill_class}">{summary}</span>', unsafe_allow_html=True
    )
    st.write("")

    total_col, passed_col, failed_col, skipped_col = st.columns(4)
    total_col.metric("Total", parsed["total"])
    passed_col.metric("Passed", counts["passed"])
    failed_col.metric("Failed", counts["failed"])
    skipped_col.metric("Skipped", counts["skipped"])

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        duration_text = (
            f"{parsed['duration']}s" if parsed["duration"] is not None else "—"
        )
        st.markdown(
            f'<div class="kv-row"><span>Execution time</span><span class="v">{duration_text}</span></div>',
            unsafe_allow_html=True,
        )
    with detail_col2:
        st.markdown(
            '<div class="kv-row"><span>Coverage</span><span class="v">Not available</span></div>',
            unsafe_allow_html=True,
        )

    if parsed["tests"]:
        st.write("")
        st.markdown(
            '<div class="card-title">Recent tests</div>', unsafe_allow_html=True
        )
        status_icons = {"PASSED": "✓", "SKIPPED": "⏭", "FAILED": "✗", "ERROR": "✗"}
        for test in parsed["tests"][:15]:
            icon = status_icons.get(test["status"], "•")
            st.markdown(f"{icon} `{html.escape(test['name'])}`")
        remaining = len(parsed["tests"]) - 15
        if remaining > 0:
            st.caption(f"...and {remaining} more")

    repair_attempts = workflow.repair_attempts_snapshot()
    latest_failure = None
    for tool_call in reversed(pytest_calls):
        if _parse_pytest_output(tool_call["output"])["exit_code"] not in (0, None):
            latest_failure = tool_call
            break

    if repair_attempts or latest_failure:
        st.write("")
        st.markdown(
            '<div class="card-title">Failure analysis</div>', unsafe_allow_html=True
        )
        for file_path, count in repair_attempts.items():
            st.caption(
                f"`{file_path}` — {count} / {workflow.MAX_REPAIR_ATTEMPTS} repair attempts"
            )
        if latest_failure:
            failed_tests = [
                test
                for test in _parse_pytest_output(latest_failure["output"])["tests"]
                if test["status"] in ("FAILED", "ERROR")
            ]
            for test in failed_tests:
                st.markdown(f"✗ `{html.escape(test['name'])}`")
            with st.expander("Raw failure output"):
                failure_output = str(latest_failure["output"])[:4000]
                st.markdown(
                    f'<pre class="log-block error">{html.escape(failure_output)}</pre>',
                    unsafe_allow_html=True,
                )

    with st.expander("Full test output"):
        latest_output = str(pytest_calls[-1]["output"])[:4000]
        st.markdown(
            f'<pre class="log-block">{html.escape(latest_output)}</pre>',
            unsafe_allow_html=True,
        )

    ruff_calls = _find_tool_calls(st.session_state.messages, "run_ruff")
    black_calls = _find_tool_calls(st.session_state.messages, "run_black")
    if ruff_calls or black_calls:
        st.write("")
        st.markdown(
            '<div class="card-title">Linting / formatting</div>', unsafe_allow_html=True
        )
        if ruff_calls:
            ruff_ok = "no issues" in str(ruff_calls[-1]["output"]).lower()
            st.markdown(
                f'<span class="pill {"success" if ruff_ok else "error"}">'
                f'Ruff — {"✓ Passed" if ruff_ok else "✗ Issues found"}</span>',
                unsafe_allow_html=True,
            )
        if black_calls:
            black_ok = not str(black_calls[-1]["output"]).lower().startswith("error")
            st.markdown(
                f'<span class="pill {"success" if black_ok else "error"}">'
                f'Black — {"✓ Passed" if black_ok else "✗ Issues found"}</span>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Git page
# ---------------------------------------------------------------------------


def render_git_page() -> None:
    st.markdown('<div class="page-title">Git</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Real, read-only Git information - nothing '
        "here stages, commits, or modifies anything.</div>",
        unsafe_allow_html=True,
    )

    if st.button("Refresh Git info", key="refresh_git_all"):
        try:
            st.session_state.git_branch_cache = tools.git_branch.invoke({})
        except Exception as exc:  # noqa: BLE001
            log_error("git_branch_ui", exc)
            st.session_state.git_branch_cache = (
                f"⚠️ Could not read the Git branch: {exc}"
            )
        try:
            st.session_state.git_status_cache = tools.git_status.invoke({})
        except Exception as exc:  # noqa: BLE001
            log_error("git_status_ui", exc)
            st.session_state.git_status_cache = (
                f"⚠️ Could not read the Git status: {exc}"
            )
        try:
            st.session_state.git_diff_cache = tools.git_diff.invoke({})
        except Exception as exc:  # noqa: BLE001
            log_error("git_diff_ui", exc)
            st.session_state.git_diff_cache = f"⚠️ Could not read the Git diff: {exc}"

    branch_cache = st.session_state.get("git_branch_cache")
    status_cache = st.session_state.get("git_status_cache")
    diff_cache = st.session_state.get("git_diff_cache")

    if branch_cache is None and status_cache is None:
        st.caption("Click Refresh to view branch, status, and diff.")
        return

    if branch_cache:
        st.markdown('<div class="card-title">Branch</div>', unsafe_allow_html=True)
        st.code(branch_cache, language=None)

    if status_cache:
        st.markdown('<div class="card-title">Status</div>', unsafe_allow_html=True)
        st.code(status_cache, language=None)

    if diff_cache:
        st.markdown('<div class="card-title">Diff</div>', unsafe_allow_html=True)
        st.markdown(_diff_to_html(diff_cache), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Logs page
# ---------------------------------------------------------------------------


def render_logs_page() -> None:
    st.markdown('<div class="page-title">Activity Logs</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Real operational events from this session '
        "(the same events printed to the terminal) - never fabricated.</div>",
        unsafe_allow_html=True,
    )

    events = get_recent_events(limit=200)
    if not events:
        st.caption("No activity logged yet this session.")
        return

    lines = [f"{event['time']}  {event['label']}  {event['text']}" for event in events]
    st.markdown(
        f'<pre class="log-block">{html.escape(chr(10).join(lines))}</pre>',
        unsafe_allow_html=True,
    )

    if st.button("Clear logs"):
        clear_events()
        st.rerun()


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------


def render_settings_page() -> None:
    st.markdown('<div class="page-title">Settings</div>', unsafe_allow_html=True)

    st.markdown('<div class="card-title">Model</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="kv-row"><span>Model</span><span class="v">{html.escape(get_model_name())}</span></div>'
        f'<div class="kv-row"><span>Temperature</span><span class="v">{AGENT_TEMPERATURE}</span></div>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        '<div class="card-title">API configuration status</div>', unsafe_allow_html=True
    )
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
        pill_class = "success" if configured else "neutral"
        state = "Configured" if configured else "Not configured"
        st.markdown(
            f'<div class="kv-row"><span>{html.escape(label)}</span>'
            f'<span class="pill {pill_class}">{state}</span></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown('<div class="card-title">Agent behavior</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="kv-row"><span>Max auto-continuation steps</span>'
        f'<span class="v">{workflow.MAX_REPAIR_ATTEMPTS} repair attempts / file</span></div>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        '<div class="card-title">Execution settings</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="kv-row"><span>Test execution</span><span class="v">Controlled '
        "(run_pytest, this project's tests/ folder only)</span></div>"
        '<div class="kv-row"><span>File writes</span><span class="v">Only after human '
        "approval</span></div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="card-title">Theme</div>', unsafe_allow_html=True)
    st.caption("Light, professional theme - matches this app's design system.")

    st.write("")
    st.markdown('<div class="card-title">Session</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="kv-row"><span>Session ID</span><span class="v">{st.session_state.session_id}</span></div>'
        f'<div class="kv-row"><span>Started at</span><span class="v">{st.session_state.started_at.strftime("%H:%M")}</span></div>',
        unsafe_allow_html=True,
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.session_state.pending_prompt = None
        st.session_state.applied_change_ids = []
        clear_events()
        st.rerun()


# ---------------------------------------------------------------------------
# Page router
# ---------------------------------------------------------------------------

_PAGE_RENDERERS = {
    "Home": render_home_page,
    "Chat": render_chat_page,
    "Project": render_project_page,
    "Documents": render_documents_page,
    "Changes": render_changes_page,
    "Tests": render_tests_page,
    "Git": render_git_page,
    "Logs": render_logs_page,
    "Settings": render_settings_page,
}

_PAGE_RENDERERS[st.session_state.nav_view]()
