"""
Tests for agent.py.

ask_agent() is tested against a fake agent object instead of the real
Gemini-backed one, so these tests never call the live API and never need a
GOOGLE_API_KEY. build_agent() is only checked for its no-API-key error path,
which is raised before any network call is made.
"""

import base64

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent as agent_module
from agent import (
    MAX_AUTO_CONTINUE_STEPS,
    SCOPE_REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    TOOLS,
    _derive_workflow_states,
    _extract_text,
    _pytest_call_passed,
    ask_agent,
    build_agent,
    classify_request,
    create_plan,
    get_api_key,
    new_ai_message,
    new_human_message,
    run_agent_turn,
    transcribe_audio,
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


# ---------------------------------------------------------------------------
# Phase 5 tool wiring and execution/error-analysis guidance
# ---------------------------------------------------------------------------


def test_check_python_syntax_tool_is_registered():
    tool_names = {t.name for t in TOOLS}
    assert "check_python_syntax" in tool_names


def test_system_prompt_mentions_check_python_syntax():
    assert "check_python_syntax" in SYSTEM_PROMPT


def test_system_prompt_covers_traceback_and_failure_analysis():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "traceback" in prompt_lower
    assert "failing test" in prompt_lower or "failure analysis" in prompt_lower
    assert "never invent a file or line number" in prompt_lower


def test_system_prompt_requires_rerun_before_claiming_fixed():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "is fixed" in prompt_lower or "is resolved" in prompt_lower
    assert "rerun" in prompt_lower or "rerun run_pytest" in prompt_lower


def test_system_prompt_covers_regression_testing():
    assert "regression" in SYSTEM_PROMPT.lower()


def test_system_prompt_rejects_arbitrary_command_execution():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "powershell" in prompt_lower
    assert "no tool for running" in prompt_lower


# ---------------------------------------------------------------------------
# Voice input (transcribe_audio) and attached-document security
# ---------------------------------------------------------------------------


class _FakeTranscriptionLLM:
    """Stands in for ChatGoogleGenerativeAI so transcribe_audio() tests never
    call the real Gemini API."""

    def __init__(self):
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return AIMessage(content="explain how tools.py works in my project")


def test_transcribe_audio_returns_transcribed_text(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    fake_llm = _FakeTranscriptionLLM()
    monkeypatch.setattr(
        agent_module, "ChatGoogleGenerativeAI", lambda **kwargs: fake_llm
    )

    text = transcribe_audio(b"fake-audio-bytes", "audio/wav")

    assert text == "explain how tools.py works in my project"


def test_transcribe_audio_sends_audio_as_base64_media_block(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    fake_llm = _FakeTranscriptionLLM()
    monkeypatch.setattr(
        agent_module, "ChatGoogleGenerativeAI", lambda **kwargs: fake_llm
    )

    transcribe_audio(b"fake-audio-bytes", "audio/wav")

    sent_message = fake_llm.received_messages[0]
    media_blocks = [
        block for block in sent_message.content if block.get("type") == "media"
    ]
    assert len(media_blocks) == 1
    assert media_blocks[0]["mime_type"] == "audio/wav"
    assert base64.b64decode(media_blocks[0]["data"]) == b"fake-audio-bytes"


def test_transcribe_audio_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        transcribe_audio(b"fake-audio-bytes")


def test_transcribe_audio_does_not_build_the_full_tool_using_agent(monkeypatch):
    # Voice input must only ever produce text - it must never spin up a
    # second, separate AI agent of its own.
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    fake_llm = _FakeTranscriptionLLM()
    monkeypatch.setattr(
        agent_module, "ChatGoogleGenerativeAI", lambda **kwargs: fake_llm
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("transcribe_audio must not build a tool-using agent")

    monkeypatch.setattr(agent_module, "create_agent", _fail_if_called)

    transcribe_audio(b"fake-audio-bytes")


def test_no_tool_can_execute_arbitrary_code_or_shell_commands():
    # Structural guarantee behind "uploaded code is never automatically
    # executed": there is no tool capable of running arbitrary code/shell
    # commands at all, uploaded or otherwise.
    forbidden_keywords = ("exec", "eval", "shell", "run_command", "run_python")
    for tool_obj in TOOLS:
        name_lower = tool_obj.name.lower()
        assert not any(
            keyword in name_lower for keyword in forbidden_keywords
        ), f"Unexpected execution-capable tool found: {tool_obj.name}"


def test_system_prompt_covers_attached_document_security():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "attached document" in prompt_lower
    assert "untrusted" in prompt_lower
    assert "never execute" in prompt_lower or "never run" in prompt_lower
    assert "arbitrary" in prompt_lower


def test_system_prompt_never_reveals_env_var_values():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "google_api_key" in prompt_lower
    assert "tavily_api_key" in prompt_lower
    assert "github_token" in prompt_lower
    assert "never state or guess their values" in prompt_lower


# ---------------------------------------------------------------------------
# General software-development questions must be in scope (not just actions
# like "generate"/"debug"/"review" applied to code the user provides).
# ---------------------------------------------------------------------------


def test_system_prompt_allows_general_conceptual_questions():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "what is python?" in prompt_lower
    assert "what is langchain?" in prompt_lower
    assert "general/conceptual questions" in prompt_lower


def test_ask_agent_answers_general_python_question_directly():
    messages = [
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a high-level programming language."),
    ]

    result = ask_agent(_FakeAgent(messages), messages)

    assert result["answer"] == "Python is a high-level programming language."
    assert result["answer"] != SCOPE_REFUSAL_MESSAGE
    assert result["tool_calls"] == []


def test_ask_agent_answers_what_is_langchain_directly():
    messages = [
        HumanMessage(content="What is LangChain?"),
        AIMessage(
            content="LangChain is a framework for building LLM-powered applications."
        ),
    ]

    result = ask_agent(_FakeAgent(messages), messages)

    assert result["answer"] != SCOPE_REFUSAL_MESSAGE
    assert result["tool_calls"] == []


def test_ask_agent_answers_explain_subprocess_directly():
    messages = [
        HumanMessage(content="Explain subprocess in Python"),
        AIMessage(content="The subprocess module lets you spawn new processes."),
    ]

    result = ask_agent(_FakeAgent(messages), messages)

    assert result["answer"] != SCOPE_REFUSAL_MESSAGE
    assert result["tool_calls"] == []


def test_ask_agent_still_refuses_unrelated_question():
    messages = [
        HumanMessage(content="Where is Japan?"),
        AIMessage(content=SCOPE_REFUSAL_MESSAGE),
    ]

    result = ask_agent(_FakeAgent(messages), messages)

    assert result["answer"] == SCOPE_REFUSAL_MESSAGE


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


# ---------------------------------------------------------------------------
# Phase 6: task planner (create_plan) and controlled-development tool wiring
# ---------------------------------------------------------------------------


class _FakePlannerLLM:
    """Stands in for ChatGoogleGenerativeAI so create_plan() tests never
    call the real Gemini API."""

    def __init__(self, response_text):
        self.response_text = response_text
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return AIMessage(content=self.response_text)


def test_create_plan_parses_numbered_steps(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    fake_llm = _FakePlannerLLM(
        "1. Inspect project structure\n"
        "2. Find related files\n"
        "3. Propose implementation\n"
        "4. Run tests\n"
    )
    monkeypatch.setattr(
        agent_module, "ChatGoogleGenerativeAI", lambda **kwargs: fake_llm
    )

    steps = create_plan("Add a login API")

    assert steps == [
        "Inspect project structure",
        "Find related files",
        "Propose implementation",
        "Run tests",
    ]


def test_create_plan_strips_bullet_and_dash_markers(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    fake_llm = _FakePlannerLLM("- Inspect project\n* Propose change\n")
    monkeypatch.setattr(
        agent_module, "ChatGoogleGenerativeAI", lambda **kwargs: fake_llm
    )

    steps = create_plan("Fix a bug")

    assert steps == ["Inspect project", "Propose change"]


def test_create_plan_ignores_blank_lines(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    fake_llm = _FakePlannerLLM("1. Step one\n\n\n2. Step two\n")
    monkeypatch.setattr(
        agent_module, "ChatGoogleGenerativeAI", lambda **kwargs: fake_llm
    )

    steps = create_plan("Refactor code")

    assert steps == ["Step one", "Step two"]


def test_create_plan_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        create_plan("Add a feature")


def test_create_plan_does_not_build_the_full_tool_using_agent(monkeypatch):
    # The planner previews a plan for the user - it must not itself spin up
    # a second, separate tool-using agent.
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    fake_llm = _FakePlannerLLM("1. Inspect project\n")
    monkeypatch.setattr(
        agent_module, "ChatGoogleGenerativeAI", lambda **kwargs: fake_llm
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("create_plan must not build a tool-using agent")

    monkeypatch.setattr(agent_module, "create_agent", _fail_if_called)

    create_plan("Add a feature")


def test_phase_6_dev_tools_are_registered():
    tool_names = {t.name for t in TOOLS}
    assert {
        "propose_file_change",
        "apply_approved_change",
        "list_pending_changes",
    } <= tool_names


def test_system_prompt_covers_phase_6_controlled_development():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "propose_file_change" in prompt_lower
    assert "apply_approved_change" in prompt_lower
    assert "approv" in prompt_lower  # "approval"/"approved"/"approve"
    assert "repair cycles" in prompt_lower or "max" in prompt_lower


def test_system_prompt_no_longer_claims_there_is_no_file_editing_tool():
    # This claim was true before Phase 6 and is now false - the system
    # prompt must not mislead the model (or the user) about this.
    assert (
        "no file-editing tool" not in SYSTEM_PROMPT.lower()
    ), "SYSTEM_PROMPT still claims there is no file-editing tool"


def test_system_prompt_has_request_classification_section():
    """Regression test for the reported bug where a development request
    containing math language (e.g. "add a function to calculate the
    factorial of a number") was wrongly routed to the calculator tool
    instead of the Phase 6 development workflow."""
    assert "## Request Classification" in SYSTEM_PROMPT
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "calculation:" in prompt_lower
    assert "development:" in prompt_lower
    assert "development + testing" in prompt_lower
    assert "development + testing + verification" in prompt_lower


def test_system_prompt_classification_warns_against_keyword_matching():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "never let one keyword" in prompt_lower
    assert "calculate the factorial of a number" in prompt_lower


def test_system_prompt_forbids_fake_verified_claims():
    """Regression test for the observed fabrication: the agent printed code
    as plain chat text (never calling propose_file_change), then called
    run_pytest against the unrelated existing suite and claimed "All tests
    passed successfully" - implying the new feature/tests had been verified
    when nothing was actually created or tested."""
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "never fabricate a" in prompt_lower
    assert "never write a fake terminal transcript" in prompt_lower
    assert "must call propose_file_change" in prompt_lower


def test_system_prompt_phase6_requires_propose_for_new_code_and_tests():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "the feature itself and any tests for it" in prompt_lower
    assert "showing code without calling propose_file_change" in prompt_lower


def test_system_prompt_forbids_truncating_modified_files():
    """Regression test for a real failure observed in live testing: the
    model proposed a "modify" of tools.py that silently cut off most of the
    file with a placeholder comment instead of reproducing it in full."""
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "must be the entire file with your change applied" in prompt_lower
    assert "rest of file omitted" in prompt_lower
    assert "relay that warning to the user verbatim" in prompt_lower


def test_system_prompt_documents_list_pending_changes():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "list_pending_changes" in prompt_lower
    assert (
        "check the real current state" in prompt_lower
        or "check for an existing" in (prompt_lower)
    )


def test_system_prompt_describes_repair_limit_as_real_not_a_guideline():
    """Regression test: MAX_REPAIR_ATTEMPTS is now a real, code-enforced
    limit (apply_approved_change refuses further writes), not just a
    system-prompt suggestion - the prompt must say so accurately."""
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "real, enforced" in prompt_lower or "real limit" in prompt_lower
    assert "not just a guideline" in prompt_lower or "not just a" in prompt_lower


# ---------------------------------------------------------------------------
# Phase 6: classify_request - dedicated development-task classification
# ---------------------------------------------------------------------------


def test_classify_request_true_for_development_response(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    fake_llm = _FakePlannerLLM("DEVELOPMENT")
    monkeypatch.setattr(
        agent_module, "ChatGoogleGenerativeAI", lambda **kwargs: fake_llm
    )

    assert (
        classify_request("Add a function to check whether a number is prime.") is True
    )


def test_classify_request_false_for_other_response(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    fake_llm = _FakePlannerLLM("OTHER")
    monkeypatch.setattr(
        agent_module, "ChatGoogleGenerativeAI", lambda **kwargs: fake_llm
    )

    assert classify_request("What is 5 factorial?") is False


def test_classify_request_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        classify_request("Add a feature")


def test_classify_request_does_not_build_the_full_tool_using_agent(monkeypatch):
    # classify_request is a single plain completion call (like create_plan) -
    # it must never itself spin up a second, separate tool-using agent.
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    fake_llm = _FakePlannerLLM("DEVELOPMENT")
    monkeypatch.setattr(
        agent_module, "ChatGoogleGenerativeAI", lambda **kwargs: fake_llm
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("classify_request must not build a tool-using agent")

    monkeypatch.setattr(agent_module, "create_agent", _fail_if_called)

    classify_request("Add a feature")


# ---------------------------------------------------------------------------
# Phase 6: run_agent_turn - the bounded auto-continuation orchestrator.
#
# Regression coverage for the reported bug: a Phase 6 development task would
# reach INSPECTING/PLANNING and then silently stop, because the underlying
# model sometimes ends its turn with only a text plan/summary and no tool
# call, which ends create_agent's tool-calling loop early. run_agent_turn
# nudges the SAME agent to keep acting instead of just narrating, bounded so
# it can never loop forever.
# ---------------------------------------------------------------------------


def _tool_call(name: str, output: str = "ok") -> dict:
    return {"name": name, "input": {}, "output": output}


def test_run_agent_turn_passes_non_development_requests_straight_through(
    monkeypatch,
):
    calls = []

    def fake_ask_agent(agent, conversation):
        calls.append(conversation)
        return {
            "answer": "4",
            "tool_calls": [_tool_call("calculator", "4")],
            "workflow_states": [],
            "pending_change_ids": [],
        }

    monkeypatch.setattr(agent_module, "ask_agent", fake_ask_agent)
    monkeypatch.setattr(agent_module, "classify_request", lambda task: False)

    conversation = [new_human_message("Calculate 2 + 2")]
    result = run_agent_turn(object(), conversation)

    assert len(calls) == 1
    assert result["answer"] == "4"


def test_run_agent_turn_stops_once_a_change_is_proposed(monkeypatch):
    calls = []

    def fake_ask_agent(agent, conversation):
        calls.append(conversation)
        return {
            "answer": "Proposed change abc123 for is_prime.py. Please approve.",
            "tool_calls": [_tool_call("propose_file_change")],
            "workflow_states": ["PROPOSING_CHANGE"],
            "pending_change_ids": ["abc123"],
        }

    monkeypatch.setattr(agent_module, "ask_agent", fake_ask_agent)
    monkeypatch.setattr(agent_module, "classify_request", lambda task: True)

    conversation = [
        new_human_message("Add a function to check whether a number is prime.")
    ]
    result = run_agent_turn(object(), conversation)

    assert len(calls) == 1  # stopped immediately - a real approval gate was reached
    assert result["pending_change_ids"] == ["abc123"]
    assert "COMPLETED" not in result["workflow_states"]


def test_run_agent_turn_nudges_a_stalled_plan_only_response_into_action(monkeypatch):
    """Regression test for the reported Phase 6 bug: the model announced a
    plan with zero tool calls and the turn ended there. This must now be
    nudged into actually proposing the change."""
    responses = [
        {
            "answer": "1. Inspect project\n2. Propose change",
            "tool_calls": [],
            "workflow_states": [],
            "pending_change_ids": [],
        },
        {
            "answer": "Proposed change xyz789. Please approve.",
            "tool_calls": [_tool_call("propose_file_change")],
            "workflow_states": ["PROPOSING_CHANGE"],
            "pending_change_ids": ["xyz789"],
        },
    ]
    calls = []

    def fake_ask_agent(agent, conversation):
        calls.append(conversation)
        return responses[len(calls) - 1]

    monkeypatch.setattr(agent_module, "ask_agent", fake_ask_agent)
    monkeypatch.setattr(agent_module, "classify_request", lambda task: True)

    conversation = [
        new_human_message("Add a function to check whether a number is prime.")
    ]
    result = run_agent_turn(object(), conversation)

    assert len(calls) == 2
    # The nudge continuation must actually be added to what the agent sees,
    # not just an identical re-send of the original conversation.
    assert len(calls[1]) > len(conversation)
    assert result["pending_change_ids"] == ["xyz789"]


def test_run_agent_turn_stops_after_two_consecutive_idle_responses(monkeypatch):
    """Never loop forever: if nudging produces no tool call twice in a row,
    give up rather than keep retrying."""
    calls = []

    def fake_ask_agent(agent, conversation):
        calls.append(conversation)
        return {
            "answer": "I'm not sure what to do next.",
            "tool_calls": [],
            "workflow_states": [],
            "pending_change_ids": [],
        }

    monkeypatch.setattr(agent_module, "ask_agent", fake_ask_agent)
    monkeypatch.setattr(agent_module, "classify_request", lambda task: True)

    conversation = [new_human_message("Add a feature.")]
    run_agent_turn(object(), conversation)

    assert len(calls) == 2  # one real attempt + one nudge, then it bails out
    assert len(calls) < MAX_AUTO_CONTINUE_STEPS


def test_run_agent_turn_never_exceeds_max_auto_continue_steps(monkeypatch):
    calls = []

    def fake_ask_agent(agent, conversation):
        calls.append(conversation)
        # Always makes "progress" (a tool call) but never actually finishes,
        # so only the hard step cap can end the loop.
        return {
            "answer": "Still working on it.",
            "tool_calls": [_tool_call("list_project_files")],
            "workflow_states": ["INSPECTING"],
            "pending_change_ids": [],
        }

    monkeypatch.setattr(agent_module, "ask_agent", fake_ask_agent)
    monkeypatch.setattr(agent_module, "classify_request", lambda task: True)

    conversation = [new_human_message("Add a feature.")]
    run_agent_turn(object(), conversation)

    assert len(calls) == MAX_AUTO_CONTINUE_STEPS


def test_run_agent_turn_stops_when_the_model_asks_a_clarifying_question(monkeypatch):
    calls = []

    def fake_ask_agent(agent, conversation):
        calls.append(conversation)
        return {
            "answer": "Which file should the new function go in?",
            "tool_calls": [],
            "workflow_states": [],
            "pending_change_ids": [],
        }

    monkeypatch.setattr(agent_module, "ask_agent", fake_ask_agent)
    monkeypatch.setattr(agent_module, "classify_request", lambda task: True)

    conversation = [new_human_message("Add a feature.")]
    result = run_agent_turn(object(), conversation)

    assert len(calls) == 1
    assert result["answer"].endswith("?")


def test_run_agent_turn_falls_back_to_ask_agent_when_classification_fails(
    monkeypatch,
):
    calls = []

    def fake_ask_agent(agent, conversation):
        calls.append(conversation)
        return {
            "answer": "ok",
            "tool_calls": [],
            "workflow_states": [],
            "pending_change_ids": [],
        }

    def failing_classify(task):
        raise RuntimeError("API hiccup")

    monkeypatch.setattr(agent_module, "ask_agent", fake_ask_agent)
    monkeypatch.setattr(agent_module, "classify_request", failing_classify)

    conversation = [new_human_message("Add a feature.")]
    result = run_agent_turn(object(), conversation)

    assert len(calls) == 1  # treated as non-development, passed straight through
    assert result["answer"] == "ok"


def test_run_agent_turn_marks_completed_after_a_passing_final_test_run(monkeypatch):
    calls = []

    def fake_ask_agent(agent, conversation):
        calls.append(conversation)
        return {
            "answer": "All tests passed.",
            "tool_calls": [_tool_call("run_pytest", "Exit code: 0\n\n5 passed")],
            "workflow_states": ["TESTING"],
            "pending_change_ids": [],
        }

    monkeypatch.setattr(agent_module, "ask_agent", fake_ask_agent)
    monkeypatch.setattr(agent_module, "classify_request", lambda task: True)

    conversation = [new_human_message("Add a feature and run the tests.")]
    result = run_agent_turn(object(), conversation)

    assert len(calls) == 1  # a passing final run is a real stopping point
    assert "REVIEWING" in result["workflow_states"]
    assert "COMPLETED" in result["workflow_states"]
    assert "FAILED" not in result["workflow_states"]


def test_run_agent_turn_marks_failed_after_a_failing_final_test_run(monkeypatch):
    def fake_ask_agent(agent, conversation):
        return {
            "answer": "Tests are still failing after repeated fixes.",
            "tool_calls": [
                _tool_call("run_pytest", "Exit code: 1\n\n1 failed, 4 passed")
            ],
            "workflow_states": ["TESTING"],
            "pending_change_ids": [],
        }

    monkeypatch.setattr(agent_module, "ask_agent", fake_ask_agent)
    monkeypatch.setattr(agent_module, "classify_request", lambda task: True)

    conversation = [new_human_message("Add a feature and run the tests.")]
    result = run_agent_turn(object(), conversation)

    assert "REVIEWING" in result["workflow_states"]
    assert "FAILED" in result["workflow_states"]
    assert "COMPLETED" not in result["workflow_states"]


def test_run_agent_turn_does_not_claim_completed_when_no_tests_ran(monkeypatch):
    def fake_ask_agent(agent, conversation):
        return {
            "answer": "Proposed the change - waiting for approval.",
            "tool_calls": [_tool_call("propose_file_change")],
            "workflow_states": ["PROPOSING_CHANGE"],
            "pending_change_ids": ["abc"],
        }

    monkeypatch.setattr(agent_module, "ask_agent", fake_ask_agent)
    monkeypatch.setattr(agent_module, "classify_request", lambda task: True)

    conversation = [new_human_message("Add a feature.")]
    result = run_agent_turn(object(), conversation)

    assert "COMPLETED" not in result["workflow_states"]
    assert "FAILED" not in result["workflow_states"]


# ---------------------------------------------------------------------------
# Phase 6: richer workflow-state derivation (PROPOSING_CHANGE, RETESTING,
# REGRESSION_TESTING, ANALYZING/FIXING) from real tool calls and run_pytest's
# own real "Exit code:" output - never fabricated.
# ---------------------------------------------------------------------------


def test_pytest_call_passed_reads_real_exit_code():
    assert _pytest_call_passed("Exit code: 0\n\n5 passed") is True
    assert _pytest_call_passed("Exit code: 1\n\n1 failed") is False
    assert _pytest_call_passed("Error: the test suite took too long to run") is None


def test_derive_workflow_states_maps_propose_to_proposing_change():
    states = _derive_workflow_states([_tool_call("propose_file_change")])
    assert states == ["PROPOSING_CHANGE"]


def test_derive_workflow_states_maps_apply_to_implementing():
    states = _derive_workflow_states([_tool_call("apply_approved_change")])
    assert states == ["IMPLEMENTING"]


def test_derive_workflow_states_first_run_pytest_is_testing():
    states = _derive_workflow_states([_tool_call("run_pytest", "Exit code: 0\n\nok")])
    assert states == ["TESTING"]


def test_derive_workflow_states_run_pytest_after_apply_is_retesting():
    states = _derive_workflow_states(
        [
            _tool_call("run_pytest", "Exit code: 1\n\n1 failed"),
            _tool_call("apply_approved_change"),
            _tool_call("run_pytest", "Exit code: 0\n\nok"),
        ]
    )
    assert states[-1] == "RETESTING"


def test_derive_workflow_states_run_pytest_without_new_apply_is_regression_testing():
    states = _derive_workflow_states(
        [
            _tool_call("run_pytest", "Exit code: 0\n\nok"),
            _tool_call("run_pytest", "Exit code: 0\n\nok"),
        ]
    )
    assert states[-1] == "REGRESSION_TESTING"


def test_derive_workflow_states_propose_after_failure_shows_analyzing_and_fixing():
    states = _derive_workflow_states(
        [
            _tool_call("run_pytest", "Exit code: 1\n\n1 failed"),
            _tool_call("propose_file_change"),
        ]
    )
    assert states == ["TESTING", "ANALYZING", "FIXING", "PROPOSING_CHANGE"]


def test_derive_workflow_states_propose_without_prior_failure_skips_analyzing():
    states = _derive_workflow_states(
        [
            _tool_call("list_project_files"),
            _tool_call("propose_file_change"),
        ]
    )
    assert states == ["INSPECTING", "PROPOSING_CHANGE"]


def test_ask_agent_includes_waiting_for_approval_in_returned_workflow_states():
    """Regression test: WAITING_FOR_APPROVAL was previously only logged to
    the terminal, never actually added to the workflow_states list the UI
    reads - so the Streamlit caption never reflected it."""
    import workflow as workflow_module

    change = workflow_module.register_change("tools.py", "create", "x = 1\n", "t")
    try:
        ai_call = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "propose_file_change", "args": {}}],
        )
        tool_result = ToolMessage(
            content=(
                f"Proposed change registered: id={change.change_id}, "
                "action=create, file=tools.py, risk=low."
            ),
            name="propose_file_change",
            tool_call_id="call_1",
        )
        final = AIMessage(
            content=f"Proposed change {change.change_id} - please approve it."
        )
        messages = [
            HumanMessage(content="Add a feature"),
            ai_call,
            tool_result,
            final,
        ]

        result = ask_agent(_FakeAgent(messages), messages)

        assert "WAITING_FOR_APPROVAL" in result["workflow_states"]
    finally:
        workflow_module.clear_all_changes()
