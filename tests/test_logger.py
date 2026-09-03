"""
Tests for logger.py.

These focus on the security-critical behavior (secrets must never reach the
terminal) and on the fact that logging never changes what ask_agent()/tools
return - only what gets printed alongside it.
"""

import io

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import ask_agent
from logger import log_error, log_tool_call, log_tool_result, sanitize
from tools import calculator


class _FakeAgent:
    def __init__(self, messages):
        self._messages = messages

    def invoke(self, _payload, config=None):
        return {"messages": self._messages}


def test_sanitize_redacts_key_value_style_secret():
    text = "GOOGLE_API_KEY=AIzaSyD-fake1234567890abcdefghijklmno"
    result = sanitize(text)
    assert "AIza" not in result
    assert "GOOGLE_API_KEY=[REDACTED]" in result


def test_sanitize_redacts_bare_google_api_key():
    text = "auth failed for key AIzaSyD-fake1234567890abcdefghijklmno during call"
    result = sanitize(text)
    assert "AIza" not in result
    assert "[REDACTED]" in result


def test_sanitize_redacts_generic_token_and_password():
    text = "token: abc123XYZ, password=hunter2"
    result = sanitize(text)
    assert "abc123XYZ" not in result
    assert "hunter2" not in result


def test_sanitize_leaves_normal_text_untouched():
    text = "125 * 48 = 6000"
    assert sanitize(text) == text


def test_log_functions_do_not_raise_and_do_not_change_return_value(capsys):
    result = calculator.invoke({"expression": "2 + 2"})
    captured = capsys.readouterr()
    assert result == "4"
    assert "[TOOL CALL]" in captured.out
    assert "Tool Name : calculator" in captured.out
    assert "[TOOL INPUT]" in captured.out
    assert "[TOOL RESULT]" in captured.out
    assert "4" in captured.out


def test_log_tool_result_survives_console_encoding_that_cannot_represent_the_text(
    monkeypatch,
):
    # Simulates Windows' default cp1252 console codepage, which cannot encode
    # many Unicode characters (e.g. CJK text, some punctuation/emoji) that can
    # legitimately show up in real web_search/documentation_search results.
    buffer = io.TextIOWrapper(
        io.BytesIO(), encoding="cp1252", errors="strict", newline=""
    )
    monkeypatch.setattr("sys.stdout", buffer)

    log_tool_result("Result contains unencodable text: 你好 — done")
    buffer.flush()

    buffer.seek(0)
    printed = buffer.buffer.getvalue().decode("cp1252")
    assert "[TOOL RESULT]" in printed
    assert "?" in printed or "REPLACEMENT" in printed.upper()


def test_log_tool_result_and_log_error_print_sanitized_content(capsys):
    log_tool_result("GOOGLE_API_KEY=AIzaSyD-fake1234567890abcdefghijklmno")
    log_error("run_pytest", "boom API_KEY=AIzaSyD-fake1234567890abcdefghijklmno")
    captured = capsys.readouterr()
    assert "AIza" not in captured.out
    assert "[TOOL RESULT]" in captured.out
    assert "[ERROR]" in captured.out


def test_ask_agent_logs_tool_decision_no_for_direct_answer(capsys):
    messages = [
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a programming language."),
    ]
    result = ask_agent(_FakeAgent(messages), messages[:1])
    captured = capsys.readouterr()

    assert result["answer"] == "Python is a programming language."
    assert "[USER INPUT]" in captured.out
    assert "Tool required: NO" in captured.out
    assert "[FINAL RESPONSE]" in captured.out


def test_ask_agent_logs_tool_decision_yes_when_tool_called(capsys):
    ai_call = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "calculator", "args": {"expression": "2+2"}}
        ],
    )
    tool_result = ToolMessage(content="4", name="calculator", tool_call_id="call_1")
    final = AIMessage(content="2 + 2 = 4")
    messages = [HumanMessage(content="Calculate 2+2"), ai_call, tool_result, final]

    result = ask_agent(_FakeAgent(messages), messages[:1])
    captured = capsys.readouterr()

    assert result["answer"] == "2 + 2 = 4"
    assert "Tool required: YES" in captured.out
    assert "Generating direct response" not in captured.out


class _FakeLLMMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class _FakeGeneration:
    def __init__(self, message):
        self.message = message


class _FakeLLMResult:
    def __init__(self, message):
        self.generations = [[_FakeGeneration(message)]]


class _CallbackAwareFakeAgent:
    """Mimics the real create_agent graph closely enough to test ordering:
    fires the on_llm_end callback (as the LangGraph "agent" node would) before
    the tool actually runs (as tools.py itself does, via log_tool_call etc.),
    exactly the sequence a real agent.invoke() produces."""

    def __init__(self, messages):
        self._messages = messages

    def invoke(self, _payload, config=None):
        callback = config["callbacks"][0]
        callback.on_llm_end(_FakeLLMResult(_FakeLLMMessage([{"id": "call_1"}])))
        log_tool_call("calculator")
        callback.on_llm_end(_FakeLLMResult(_FakeLLMMessage([])))
        return {"messages": self._messages}


def test_tool_decision_is_logged_before_tool_call_for_a_real_agent(capsys):
    ai_call = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "calculator", "args": {"expression": "2+2"}}
        ],
    )
    tool_result = ToolMessage(content="4", name="calculator", tool_call_id="call_1")
    final = AIMessage(content="2 + 2 = 4")
    messages = [HumanMessage(content="Calculate 2+2"), ai_call, tool_result, final]

    ask_agent(_CallbackAwareFakeAgent(messages), messages[:1])
    captured = capsys.readouterr()

    decision_index = captured.out.index("[TOOL DECISION]")
    tool_call_index = captured.out.index("[TOOL CALL]")
    assert decision_index < tool_call_index
