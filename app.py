"""
app.py
------
This file is ONLY responsible for the Streamlit user interface: showing the
chat, taking user input, and displaying the agent's answers. All the AI logic
lives in agent.py, and the tools live in tools.py.

    USER -> STREAMLIT UI -> agent.py (LangChain + Gemini) -> back to UI
"""

import html
import re

import streamlit as st
import streamlit.components.v1 as components

import documents
from agent import (
    ask_agent,
    build_agent,
    get_api_key,
    new_ai_message,
    new_human_message,
    transcribe_audio,
)
from logger import log_error

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
    .block-container { padding-top: 2rem; }
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
    </style>
    """,
    unsafe_allow_html=True,
)

# AI Features reference panel (sidebar): pure information about what each
# phase can do - no example question is ever inserted into the chat input or
# sent to the agent from here. The user always types their own question into
# the existing chat bar; this dict only feeds read-only st.markdown() text.
AI_FEATURES_INFO = {
    "Phase 1 — Foundation & Basic AI Agent": {
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
    else:
        body = _output_block(tool_call["output"])

    st.markdown(
        f'<div class="tool-box"><b>{header}</b><br>{body}</div>', unsafe_allow_html=True
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🤖 AI Developer Assistant")
    st.caption("Your AI coding companion.")

    st.markdown("## ✨ Features")
    st.markdown(
        "✓ Programming Q&A\n\n"
        "✓ Code Generation\n\n"
        "✓ Code Explanation\n\n"
        "✓ Debugging\n\n"
        "✓ Code Review\n\n"
        "✓ Code Refactoring\n\n"
        "✓ Test Generation\n\n"
        "✓ Pytest\n\n"
        "✓ Ruff\n\n"
        "✓ Black\n\n"
        "✓ Syntax Checking\n\n"
        "✓ Test Failure & Traceback Analysis\n\n"
        "✓ Project Awareness\n\n"
        "✓ Web & Documentation Search\n\n"
        "✓ Git Tools (read-only)\n\n"
        "✓ GitHub Issues/PRs\n\n"
        "✓ Commit Message Generation\n\n"
        "✓ 🎤 Voice Assistant\n\n"
        "✓ 📎 Document Analysis"
    )

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
    if st.button("🗑 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.session_state.pending_prompt = None
        st.rerun()

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown("# 🤖 AI Developer Assistant")
with header_right:
    st.markdown("### 🟢 Online")

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
# Empty state
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.markdown("### 👋 Welcome!")
    st.markdown(
        "I'm your AI Developer Assistant. I can help you with:\n\n"
        "💡 Programming Q&A · 🧮 Calculations · 🐍 Code Explanation\n\n"
        "✍️ Code Generation · 🐞 Debugging · 🔍 Code Review · ♻️ Refactoring\n\n"
        "🧪 Test Generation · ✅ Pytest · 🔎 Ruff · 🎨 Black\n\n"
        "🎤 Voice Questions · 📎 Document Upload (code, text, PDF, DOCX)\n\n"
        "Pick a **Quick Action** in the sidebar, or just ask naturally - "
        'try *"Write a Python function to check whether a number is prime."*'
    )

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        for tool_call in message.get("tool_calls", []):
            render_tool_box(tool_call)
        if message["role"] == "assistant":
            render_answer(message["content"])
            render_tts_button(message["content"], key=f"tts_history_{index}")
        else:
            if message.get("source") == "voice":
                st.caption("🎤 Voice input · 📝 Transcribed")
            st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Handle new input: typed text, an attached document, a recorded voice
# question, or a prompt clicked from the sidebar - all flow into the exact
# same agent call below. Voice is only ever converted to text (transcribe_audio)
# before reaching the agent; attached documents are only ever added as labeled,
# untrusted reference context - neither path adds a separate AI implementation.
# ---------------------------------------------------------------------------

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
                    chat_value.audio.getvalue(), chat_value.audio.type or "audio/wav"
                )
            except Exception as exc:  # noqa: BLE001 - never let STT crash the chat
                log_error("transcribe_audio", exc)
                st.error(
                    "🎤 Could not transcribe your voice input. Please try again, "
                    "or continue using the text input."
                )
        if voice_text:
            st.info(f'📝 Transcribed: "{voice_text}"')
        elif chat_value.audio is not None:
            st.warning("🎤 No speech was detected in that recording. Please try again.")

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

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
            "tool_calls": [],
            "source": input_source,
        }
    )
    st.session_state.lc_history.append(new_human_message(agent_input))

    with st.chat_message("user"):
        if input_source == "voice":
            st.caption("🎤 Voice input · 📝 Transcribed")
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask_agent(agent, st.session_state.lc_history)
                answer = result["answer"]
                tool_calls = result["tool_calls"]
            except Exception as exc:  # noqa: BLE001
                print(f"[agent error] {exc}")
                answer = "⚠️ I couldn't process that request. Please check your API key or try again."
                tool_calls = []

        for tool_call in tool_calls:
            render_tool_box(tool_call)

        render_answer(answer)
        render_tts_button(answer, key=f"tts_live_{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tool_calls": tool_calls}
    )
    st.session_state.lc_history.append(new_ai_message(answer))
