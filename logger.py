"""
logger.py
---------
Developer-facing terminal logging for the AI Developer Assistant.

This module ONLY prints observable execution events (what the agent actually
did) - never hidden reasoning or chain-of-thought, and never secrets. It is
called from:
    - agent.py  -> user input, agent start, tool decision, LLM/final response
    - tools.py  -> tool call, tool input, tool execution, tool result, errors

Every function here is a small, focused print helper so call sites stay a
single line each instead of scattering ad-hoc print statements everywhere.
"""

import re
import sys
import time
from collections import deque

SEPARATOR = "=" * 60
_INPUT_TRUNCATE_LIMIT = 1500
_RESULT_TRUNCATE_LIMIT = 3000

# ---------------------------------------------------------------------------
# In-memory event buffer: the Streamlit UI's "Agent Logs" panel has no way
# to read this process's own real stdout, so every _section() call below
# (already sanitized/truncated - the exact same content that prints to the
# terminal) also lands here as a timestamped entry the UI can read back with
# get_recent_events(). Single-user simplification: a process-wide buffer,
# not per-session, matching workflow.py's pending-change registry - this
# project runs as one local, single-user app/CLI rather than a multi-tenant
# service.
# ---------------------------------------------------------------------------

_MAX_EVENT_ENTRIES = 500
_events: deque = deque(maxlen=_MAX_EVENT_ENTRIES)


def get_recent_events(limit: int = 200) -> list[dict]:
    """The most recent `limit` logged events, oldest first - each a plain
    {"time": "HH:MM:SS", "label": str, "text": str} dict, already the same
    sanitized/truncated text that was printed to the terminal."""
    events = list(_events)
    return events[-limit:] if limit else events


def clear_events() -> None:
    """Discard the buffered event history (e.g. when the user clears the
    conversation) - never affects what was already printed to the terminal."""
    _events.clear()


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------
# Defense in depth: even though no code path here logs .env contents or raw
# environment variables, any text passed in (tool output, error messages,
# user-pasted code) is scrubbed for anything that looks like a credential
# before it is ever printed.

_KEY_VALUE_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)[A-Z0-9_]*)"
    r"\s*[:=]\s*[^\s,;\"']+"
)
_GOOGLE_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b")
_GITHUB_TOKEN_RE = re.compile(
    r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[0-9A-Za-z_]{20,}\b"
)
_TAVILY_KEY_RE = re.compile(r"\btvly-[0-9A-Za-z_\-]{10,}\b")


def sanitize(text) -> str:
    """Redact anything that looks like an API key/token/secret/password."""
    text = str(text)
    text = _KEY_VALUE_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = _GOOGLE_KEY_RE.sub("[REDACTED]", text)
    text = _GITHUB_TOKEN_RE.sub("[REDACTED]", text)
    text = _TAVILY_KEY_RE.sub("[REDACTED]", text)
    return text


def _truncate(text: str, limit: int) -> str:
    text = str(text).strip()
    if len(text) > limit:
        return text[:limit] + "\n... (truncated)"
    return text


def _safe_print(text: str) -> None:
    """Print text that may contain characters the terminal's codepage can't
    encode (e.g. Windows' default cp1252 console hitting non-Latin/emoji
    characters pulled from web_search/documentation_search results).

    Without this, an otherwise-successful tool call could crash the whole
    request with a UnicodeEncodeError purely because of what a web page
    happened to contain - printing must never be why a request fails.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(
            text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        )


def _section(label: str, body: str) -> None:
    print()
    print(f"[{label}]")
    _safe_print(body)
    _events.append({"time": time.strftime("%H:%M:%S"), "label": label, "text": body})


# ---------------------------------------------------------------------------
# High-level flow (called from agent.py)
# ---------------------------------------------------------------------------


def log_request_start() -> None:
    print(SEPARATOR)
    print("AI DEVELOPER ASSISTANT")
    print(SEPARATOR)


def log_user_input(text: str) -> None:
    _section("USER INPUT", _truncate(sanitize(text), _INPUT_TRUNCATE_LIMIT))


def log_phase6_task_received(task: str) -> None:
    _section(
        "PHASE 6", f"Task received: {_truncate(sanitize(task), _INPUT_TRUNCATE_LIMIT)}"
    )


def log_planner_result(steps: list[str]) -> None:
    body = (
        "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))
        or "(no plan)"
    )
    _section("PLANNER", body)


def log_workflow_state(state: str) -> None:
    _section("WORKFLOW", f"State: {state}")


def log_approval_waiting(change_id: str, file_path: str) -> None:
    _section(
        "APPROVAL",
        f"Waiting for user approval - change_id={change_id} file={file_path}",
    )


def log_agent_start() -> None:
    _section("AGENT", "Processing request...")


def log_tool_decision(tool_required: bool) -> None:
    _section("TOOL DECISION", f"Tool required: {'YES' if tool_required else 'NO'}")


def log_llm_direct_response() -> None:
    _section("LLM", "Generating direct response...")


def log_final_response(text: str) -> None:
    _section("FINAL RESPONSE", _truncate(sanitize(text), _RESULT_TRUNCATE_LIMIT))
    print()
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Tool-level (called from tools.py, at the point each tool actually runs)
# ---------------------------------------------------------------------------


def log_tool_call(name: str) -> None:
    _section("TOOL CALL", f"Tool Name : {name}")


def log_tool_input(text: str) -> None:
    _section("TOOL INPUT", _truncate(sanitize(text), _INPUT_TRUNCATE_LIMIT))


def log_tool_execution(message: str) -> None:
    _section("TOOL EXECUTION", message)


def log_tool_result(text: str) -> None:
    _section("TOOL RESULT", _truncate(sanitize(text), _RESULT_TRUNCATE_LIMIT))


def log_exit_code(code: int) -> None:
    _section("EXIT CODE", str(code))


def log_error(tool_name: str, error) -> None:
    _section(
        "ERROR",
        f"Tool execution failed.\n\nTool Name : {tool_name}\nError     : {sanitize(error)}",
    )
