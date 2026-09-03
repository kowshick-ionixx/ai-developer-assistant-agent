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

from agent import ask_agent, build_agent, get_api_key, new_ai_message, new_human_message

EXIT_COMMANDS = {"exit", "quit", "q"}


def main() -> None:
    api_key = get_api_key()
    if not api_key or api_key == "your_google_api_key_here":
        print(
            "GOOGLE_API_KEY is not configured. Please add it to your .env file "
            "(see .env.example) and try again."
        )
        return

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

        try:
            result = ask_agent(agent, conversation)
            answer = result["answer"]
        except Exception as exc:  # noqa: BLE001 - keep the CLI session alive
            from logger import log_error

            log_error("ask_agent", exc)
            answer = "I couldn't process that request. Please check your API key or try again."

        conversation.append(new_ai_message(answer))
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
