"""
Tests for agent.py.

ask_agent() is tested against a fake agent object instead of the real
Gemini-backed one, so these tests never call the live API and never need a
GOOGLE_API_KEY. build_agent() is only checked for its no-API-key error path,
which is raised before any network call is made.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import (
    SCOPE_REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    TOOLS,
    _extract_text,
    ask_agent,
    build_agent,
    get_api_key,
    new_ai_message,
    new_human_message,
)


class _FakeAgent:
    """Stands in for the compiled LangChain agent so tests never call Gemini."""

    def __init__(self, messages):
        self._messages = messages

    def invoke(self, _payload, config=None):
        return {"messages": self._messages}


def test_get_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    assert get_api_key() == "test-key-123"


def test_build_agent_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        build_agent()


def test_new_human_message():
    message = new_human_message("hello")
    assert isinstance(message, HumanMessage)
    assert message.content == "hello"


def test_new_ai_message():
    message = new_ai_message("hi there")
    assert isinstance(message, AIMessage)
    assert message.content == "hi there"


def test_extract_text_from_plain_string():
    assert _extract_text("plain text") == "plain text"


def test_extract_text_from_content_block_list():
    content = [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
    assert _extract_text(content) == "hello world"


def test_ask_agent_direct_answer_has_no_tool_calls():
    messages = [
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a programming language."),
    ]

    result = ask_agent(_FakeAgent(messages), messages)

    assert result["answer"] == "Python is a programming language."
    assert result["tool_calls"] == []


def test_scope_refusal_message_is_exact_expected_sentence():
    assert SCOPE_REFUSAL_MESSAGE == (
        "I can only assist with tasks related to this AI Developer Assistant "
        "project and software development."
    )


def test_system_prompt_instructs_scope_refusal_with_exact_message():
    assert SCOPE_REFUSAL_MESSAGE in SYSTEM_PROMPT
    assert "## Scope" in SYSTEM_PROMPT
    assert "## Out-of-scope questions" in SYSTEM_PROMPT


def test_ask_agent_extracts_tool_call_name_input_and_output():
    ai_call = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "calculator", "args": {"expression": "2+2"}}
        ],
    )
    tool_result = ToolMessage(content="4", name="calculator", tool_call_id="call_1")
    final = AIMessage(content="2 + 2 = 4")
    messages = [HumanMessage(content="Calculate 2+2"), ai_call, tool_result, final]

    result = ask_agent(_FakeAgent(messages), messages)

    assert result["answer"] == "2 + 2 = 4"
    assert result["tool_calls"] == [
        {"name": "calculator", "input": {"expression": "2+2"}, "output": "4"}
    ]


# ---------------------------------------------------------------------------
# Phase 4 tool wiring
# ---------------------------------------------------------------------------

_PHASE_4_TOOL_NAMES = {
    "web_search",
    "documentation_search",
    "git_status",
    "git_log",
    "git_diff",
    "git_branch",
    "github_get_repository",
    "github_get_issues",
    "github_get_pull_requests",
}


def test_phase_4_tools_are_registered():
    tool_names = {t.name for t in TOOLS}
    assert _PHASE_4_TOOL_NAMES <= tool_names


def test_phase_1_to_3_tools_still_registered():
    tool_names = {t.name for t in TOOLS}
    original_tools = {
        "calculator",
        "explain_python_code",
        "run_pytest",
        "run_ruff",
        "run_black",
        "list_project_files",
        "read_project_file",
        "search_project",
    }
    assert original_tools <= tool_names


def test_system_prompt_mentions_phase_4_tools_and_untrusted_web_content():
    for name in _PHASE_4_TOOL_NAMES:
        assert name in SYSTEM_PROMPT
    assert "untrusted" in SYSTEM_PROMPT.lower()
    assert (
        "never actually run" in SYSTEM_PROMPT.lower()
        or "never run" in SYSTEM_PROMPT.lower()
    )


def test_ask_agent_extracts_git_diff_tool_call():
    diff_output = "diff --git a/agent.py b/agent.py\n+added line\n"
    ai_call = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "git_diff", "args": {"file_path": ""}}],
    )
    tool_result = ToolMessage(
        content=diff_output, name="git_diff", tool_call_id="call_1"
    )
    final = AIMessage(content="feat: update agent.py")
    messages = [
        HumanMessage(content="Create a commit message from my changes."),
        ai_call,
        tool_result,
        final,
    ]

    result = ask_agent(_FakeAgent(messages), messages)

    assert result["answer"] == "feat: update agent.py"
    assert result["tool_calls"] == [
        {"name": "git_diff", "input": {"file_path": ""}, "output": diff_output}
    ]
