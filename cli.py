"""
cli.py
------
A plain terminal chat interface for the AI Developer Assistant, as an
alternative to the Streamlit app (app.py). It reuses the exact same
agent.py/tools.py logic - no separate AI wiring - so behavior (and the
detailed [USER INPUT]/[TOOL CALL]/[FINAL RESPONSE] terminal logging from
logger.py) is identical to what you'd get running the app through Streamlit.

Usage:
    python cli.py

Type "exit", "quit", or press Ctrl+C to stop.
"""

import workflow
from agent import (
    api_key_looks_valid,
    build_agent,
    describe_agent_error,
    get_api_key,
    new_ai_message,
    new_human_message,
    run_agent_turn,
)
from logger import safe_print

EXIT_COMMANDS = {"exit", "quit", "q"}


def _print_pending_change(change: workflow.ProposedChange) -> None:
    """Show one proposed change's full detail - the CLI's equivalent of the
    Streamlit sidebar's expandable "Pending Approvals" card.

    Uses safe_print (not a bare print()) for every field that can hold
    AI-generated free text (`reason`, and especially `content` - a whole
    proposed file's text, which regularly contains emoji, e.g. a Streamlit
    app's own UI copy). Confirmed live: a bare print() of generated file
    content crashed the entire CLI session with UnicodeEncodeError on a
    legacy Windows console codepage (cp1252) - the exact same class of bug
    already fixed for the "Assistant: ..." answer line (see
    test_run_turn_does_not_crash_on_a_legacy_console_codepage), just missed
    here."""
    safe_print(f"\n{'=' * 60}")
    safe_print("APPROVAL REQUIRED")
    safe_print(f"{'=' * 60}")
    safe_print(f"File   : {change.file_path}")
    safe_print(f"Action : {change.action}")
    safe_print(f"Risk   : {change.risk}")
    safe_print(f"Reason : {change.reason}")
    safe_print(f"{'-' * 60}")
    safe_print(change.content)
    safe_print(f"{'=' * 60}")


def _handle_pending_approvals(change_ids: list[str]) -> list[str]:
    """Prompt the user to approve/reject each pending change_id in the
    terminal - the CLI's equivalent of the Streamlit sidebar's Approve/
    Reject buttons. workflow.approve_change()/reject_change() are only ever
    called from here or from app.py's UI, never from a tool the agent
    itself can call. Returns the change_ids that were actually approved."""
    approved_ids = []
    for change_id in change_ids:
        change = workflow.get_change(change_id)
        if change is None or change.applied:
            continue
        _print_pending_change(change)
        try:
            answer = input(f"Approve this change ({change_id})? [y/N]: ")
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer.strip().lower() in ("y", "yes"):
            workflow.approve_change(change_id)
            approved_ids.append(change_id)
            print(f"Approved {change_id}.\n")
        else:
            workflow.reject_change(change_id)
            print(f"Rejected {change_id} - no changes were made.\n")
    return approved_ids


# A safety cap on how many approve-then-continue rounds a single user
# message can trigger (e.g. propose -> approve -> apply -> test fails ->
# propose a fix -> approve -> ...). Mirrors workflow.MAX_REPAIR_ATTEMPTS
# with headroom for the initial proposal, so this can never loop forever
# even if the model kept proposing new fixes past the repair-attempt limit.
_MAX_APPROVAL_ROUNDS = workflow.MAX_REPAIR_ATTEMPTS + 2


def _run_turn(agent, conversation: list) -> str:
    """Run one agent turn, print/record the reply, and - the CLI's own
    equivalent of the Streamlit sidebar's Approve/Reject buttons - resolve
    any Phase 6 approval pause point(s) that come up along the way,
    including a fix's own follow-up proposal after a failing test run.
    Returns the final answer shown to the user."""
    try:
        result = run_agent_turn(agent, conversation)
        answer = result["answer"]
    except Exception as exc:  # noqa: BLE001 - keep the CLI session alive
        from logger import log_error

        log_error("run_agent_turn", exc)
        answer = describe_agent_error(exc)
        conversation.append(new_ai_message(answer))
        safe_print(f"\nAssistant: {answer}\n")
        return answer

    conversation.append(new_ai_message(answer))
    safe_print(f"\nAssistant: {answer}\n")

    pending_change_ids = result.get("pending_change_ids") or []
    for _ in range(_MAX_APPROVAL_ROUNDS):
        if not pending_change_ids:
            break

        approved_ids = _handle_pending_approvals(pending_change_ids)
        if not approved_ids:
            print("No changes were approved - nothing was written to disk.\n")
            return answer

        follow_up = (
            "The following change(s) have been approved: "
            f"{', '.join(approved_ids)}. Please apply them, run the complete "
            "test suite, fix any failures if necessary, and verify existing "
            "functionality still works."
        )
        conversation.append(new_human_message(follow_up))
        try:
            result = run_agent_turn(agent, conversation)
            answer = result["answer"]
        except Exception as exc:  # noqa: BLE001 - keep the CLI session alive
            from logger import log_error

            log_error("run_agent_turn", exc)
            answer = describe_agent_error(exc)
            conversation.append(new_ai_message(answer))
            safe_print(f"\nAssistant: {answer}\n")
            return answer

        conversation.append(new_ai_message(answer))
        safe_print(f"\nAssistant: {answer}\n")
        pending_change_ids = result.get("pending_change_ids") or []

    return answer


def main() -> None:
    api_key = get_api_key()
    if not api_key or not api_key.strip() or api_key == "your_google_api_key_here":
        print(
            "GOOGLE_API_KEY is not configured. Please add it to your .env file "
            "(see .env.example) and try again."
        )
        return
    if not api_key_looks_valid(api_key):
        print(
            "Note: GOOGLE_API_KEY doesn't match the traditional 'AIza...' "
            "Gemini key format, but continuing - Google's API will "
            "determine whether it's actually valid.\n"
        )

    try:
        agent = build_agent()
    except ValueError as exc:
        print(f"Could not start the AI agent: {exc}")
        return

    print("AI Developer Assistant (CLI mode). Type 'exit' to quit.\n")

    conversation: list = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("Goodbye!")
            break

        conversation.append(new_human_message(user_input))
        _run_turn(agent, conversation)


if __name__ == "__main__":
    main()
