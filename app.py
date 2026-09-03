"""
app.py
------
This file is ONLY responsible for the Streamlit user interface: showing the
chat, taking user input, and displaying the agent's answers. All the AI logic
lives in agent.py, and the tools live in tools.py.

    USER -> STREAMLIT UI -> agent.py (LangChain + Gemini) -> back to UI
"""

import html

import streamlit as st

from agent import ask_agent, build_agent, get_api_key, new_ai_message, new_human_message
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
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLE_QUESTIONS = [
    "What is Python?",
    "Calculate 125 * 48",
    "Explain this Python code:\n\nfor i in range(5):\n    print(i)",
    "Write a Python function to check whether a number is prime.",
    "Review this code:\n\ndef add_numbers(a,b):\n return a+b",
    "Run the tests.",
    "What is this project?",
    "What files have changed in my project?",
    "Create a commit message from my current changes.",
    "Find the official documentation for Python pathlib.",
]

TOOL_DISPLAY_NAMES = {
    "calculator": "🔧 Using Calculator Tool",
    "explain_python_code": "🔧 Using Code Explanation Tool",
    "run_pytest": "🔧 Running Pytest",
    "run_ruff": "🔧 Running Ruff",
    "run_black": "🔧 Running Black",
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
        "✓ Project Awareness\n\n"
        "✓ Web & Documentation Search\n\n"
        "✓ Git Tools (read-only)\n\n"
        "✓ GitHub Issues/PRs\n\n"
        "✓ Commit Message Generation"
    )

    st.markdown("## 🎯 Quick Actions")
    mode = st.selectbox(
        "Mode",
        list(MODE_PREFIXES.keys()),
        key="mode_select",
        label_visibility="collapsed",
        help="Choose what kind of help you need, then describe or paste your code below.",
    )

    st.markdown("## 💬 Example Questions")
    for question in EXAMPLE_QUESTIONS:
        label = question.split("\n")[0]
        if st.button(label, key=f"example_{label}", use_container_width=True):
            st.session_state.pending_prompt = question

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
        "Pick a **Quick Action** in the sidebar, or just ask naturally - "
        'try *"Write a Python function to check whether a number is prime."*'
    )

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        for tool_call in message.get("tool_calls", []):
            render_tool_box(tool_call)
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Handle new input (either typed, or clicked from the sidebar examples)
# ---------------------------------------------------------------------------

user_input = st.chat_input("Ask something...")
if st.session_state.pending_prompt and not user_input:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_input:
    user_input = user_input.strip()

if user_input:
    # The chat displays what the user actually typed; the agent additionally
    # receives the Quick Action prefix (if any) to clarify intent.
    prefix = MODE_PREFIXES.get(st.session_state.get("mode_select", "Ask"), "")
    agent_input = f"{prefix}{user_input}" if prefix else user_input

    st.session_state.messages.append(
        {"role": "user", "content": user_input, "tool_calls": []}
    )
    st.session_state.lc_history.append(new_human_message(agent_input))

    with st.chat_message("user"):
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

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tool_calls": tool_calls}
    )
    st.session_state.lc_history.append(new_ai_message(answer))
