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

# Deliberately stricter than _KEY_VALUE_RE above: requires the assigned
# value to itself be a quoted, sufficiently long literal (e.g.
# `API_KEY = "abcd1234..."'`), not just any assignment to a plausibly-named
# variable. `_KEY_VALUE_RE` is tuned for redacting already-suspicious tool
# output/pasted text, where matching on the name alone is the right
# trade-off; scanning arbitrary real source code with that same loose
# pattern would flag completely ordinary code such as
# `api_key = os.getenv("GOOGLE_API_KEY")` or `token = get_token()` (the
# value is a call/expression, never a quoted literal) - see
# contains_probable_secret's docstring.
_HARDCODED_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|secret|token|password|credential)\w*"
    r"\s*[:=]\s*[\"']([A-Za-z0-9_\-]{16,})[\"']"
)


def sanitize(text) -> str:
    """Redact anything that looks like an API key/token/secret/password."""
    text = str(text)
    text = _KEY_VALUE_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = _GOOGLE_KEY_RE.sub("[REDACTED]", text)
    text = _GITHUB_TOKEN_RE.sub("[REDACTED]", text)
    text = _TAVILY_KEY_RE.sub("[REDACTED]", text)
    return text


def contains_probable_secret(text) -> bool:
    """True if `text` contains something that looks like an actual hardcoded
    secret VALUE: a known API key/token format (Google/GitHub/Tavily), or a
    KEY/SECRET/TOKEN/PASSWORD/CREDENTIAL assignment whose value is itself a
    quoted, sufficiently long literal.

    Used as a content-level defense-in-depth check (e.g. tools.py's project
    ZIP packaging scans real file contents before archiving them) so a
    secret accidentally left in a non-obviously-named file isn't missed by a
    filename-only exclusion list. Deliberately does NOT reuse sanitize()'s
    looser _KEY_VALUE_RE here - that pattern matches on the variable name
    alone (right for redacting already-suspicious tool output/pasted text),
    which would misfire on completely ordinary source code that merely
    assigns a variable named api_key/token/secret to a non-literal
    expression (e.g. `api_key = os.getenv("GOOGLE_API_KEY")`,
    `token = get_token()`) - exactly the kind of code this project's own
    files legitimately contain. Never itself reveals the matched value -
    callers only ever get True/False back."""
    text = str(text)
    return bool(
        _GOOGLE_KEY_RE.search(text)
        or _GITHUB_TOKEN_RE.search(text)
        or _TAVILY_KEY_RE.search(text)
        or _HARDCODED_SECRET_VALUE_RE.search(text)
    )


def _truncate(text: str, limit: int) -> str:
    text = str(text).strip()
    if len(text) > limit:
        return text[:limit] + "\n... (truncated)"
    return text


def _safe_print(text: str) -> None:
    """Print text that may contain characters the terminal's codepage can't
    encode (e.g. Windows' default cp1252 console hitting non-Latin/emoji
    characters pulled from web_search/documentation_search results, or from
    Gemini's own answer text, which routinely includes emoji).

    Without this, an otherwise-successful tool call (or, via safe_print()
    below, a successful chat turn) could crash the whole request with a
    UnicodeEncodeError purely because of what the text happened to contain -
    printing must never be why a request fails.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(
            text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        )


def safe_print(text: str) -> None:
    """Public entry point to _safe_print() for callers outside this module
    (e.g. cli.py's "Assistant: ..." lines) that print model-generated text
    directly to the console and need the same protection - see _safe_print's
    docstring. Confirmed live: a real Gemini answer containing a folder emoji
    crashes a bare print() with UnicodeEncodeError on a legacy Windows
    console codepage (cp1252)."""
    _safe_print(text)


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


# ---------------------------------------------------------------------------
# Performance timing (debugging aid only)
# ---------------------------------------------------------------------------
# Deliberately NOT routed through _section()/_events - these lines are for a
# developer watching the terminal to see where time is actually going
# (classification, LLM calls, tool calls, test runs, ...), not for the
# Streamlit "Agent Logs" panel, so they never clutter what an end user sees.


def log_perf(label: str, seconds: float) -> None:
    """Print a single lightweight "[PERF] <label>: <seconds>s" terminal line."""
    _safe_print(f"[PERF] {label}: {seconds:.2f}s")


# ---------------------------------------------------------------------------
# LLM call instrumentation (quota/rate-limit observability)
# ---------------------------------------------------------------------------
# Unlike log_perf() above, this IS routed through _section() - so it also
# lands in get_recent_events() for the Streamlit "Agent Logs" panel, not just
# the terminal. That matters here specifically: distinguishing "one real
# Gemini request" from "several silently retried requests" (and seeing which
# ones failed with what category) is exactly the visibility needed to tell a
# real Google quota problem apart from the app making excessive calls, and a
# developer watching only the Streamlit UI (not raw stdout) needs to see it
# too. Only ever safe, non-secret metadata - never the request/response
# content, the API key, or any header.


def log_llm_call(
    *,
    purpose: str,
    model: str,
    attempt: int,
    duration: float,
    status: str,
    status_category: str | None = None,
    detail: str | None = None,
) -> None:
    """Log one real Gemini request attempt: what it was for, which model,
    which attempt number (for retries), how long it took, and whether it
    succeeded. `status` is "success" or "failure"; `status_category` further
    classifies a failure (e.g. "rate_limit", "quota_exhausted", "timeout",
    "invalid_credential") for retry-handling and quota-vs-bug diagnosis.
    `detail` is optional extra safe context (e.g. the specific Google quota
    metric name a 429 named) - never raw exception text, which sanitize()
    would be required for."""
    lines = [
        f"Purpose: {purpose}",
        f"Model: {model}",
        f"Attempt: {attempt}",
        f"Duration: {duration:.2f}s",
        f"Status: {status}",
    ]
    if status_category:
        lines.append(f"Category: {status_category}")
    if detail:
        lines.append(f"Detail: {sanitize(detail)}")
    _section("LLM", "\n".join(lines))
