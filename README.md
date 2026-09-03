# 🤖 AI Developer Assistant

## Overview

A beginner-friendly AI agent built with **Python**, **LangChain**, **Google Gemini**, and **Streamlit** that behaves like a junior software developer. It answers programming questions, does math, explains code, and — as of Phase 2 — generates, debugs, reviews, refactors, and tests code, and can run Pytest/Ruff/Black on request. It decides for itself, on every message, whether it needs a tool or can just answer directly. It is also **project-aware**: it can inspect this project's own real structure, files, and source code (via `list_project_files`/`read_project_file`/`search_project`) to answer questions about itself accurately, instead of guessing. As of Phase 4, it can also reach outside the project: **web/documentation search** (via Tavily) for current framework/API information, **read-only Git inspection**, **read-only GitHub repository/issue/PR lookups**, and **commit-message and documentation generation** grounded in the project's real diff/files.

This project exists to teach one thing clearly: **how an AI agent actually works**.

## Features

**Phase 1**
- 💡 Programming Q&A (answered directly by Gemini)
- 🧮 Calculator (safe expression evaluation — no `eval()`)
- 🐍 Python code explanation (structural analysis — code is never executed)

**Phase 2**
- ✍️ Code generation (Explanation / Code / Usage)
- 🐞 Debugging (Problem / Cause / Fixed Code / Explanation)
- 🔍 Code review (Issues Found / Suggestions / Improved Code)
- ♻️ Refactoring (Original Problem / Refactored Code / Improvements)
- 🧪 Pytest test generation
- ✅ Running the project's test suite with Pytest
- 🔎 Code quality checks with Ruff
- 🎨 Code formatting checks with Black

**Project awareness (Phase 2 -> 3 transition)**
- 🗂️ Real project structure inspection (`list_project_files`)
- 📖 Real project file reading (`read_project_file`)
- 🔍 Real project-wide search for functions/classes/text (`search_project`)
- Accurate answers to questions like *"what is this project?"*, *"what does tools.py do?"*, or *"how does run_ruff work?"*, grounded in the project's actual files instead of guesses

**Phase 4**
- 🌐 Web search (`web_search`) for current, general technical information
- 📚 Official documentation search (`documentation_search`) — biased toward authoritative sources (`docs.python.org`, `python.langchain.com`, etc.) when the topic is recognized
- 🧭 Current-API awareness — the assistant verifies current framework/API details with a tool instead of presenting outdated training knowledge as current, and says so when it can't verify
- 🔧 Read-only Git inspection: `git_status`, `git_log`, `git_diff`, `git_branch`
- 🐙 Read-only GitHub lookups: `github_get_repository`, `github_get_issues`, `github_get_pull_requests`
- 📝 Commit-message generation — reads the real `git_diff` output and writes a conventional-style message (never runs `git commit`)
- 📄 Documentation generation — inspects real project files (`read_project_file`/`search_project`) and writes documentation as its answer
- 🧩 Combined reasoning — compares this project's real code against current official documentation and labels which parts of the answer come from each

**Both phases**
- 🔧 Visible tool calling, so you can see exactly when and why a tool ran
- 🎯 Optional Quick Action mode selector (Ask / Generate / Debug / Review / Refactor / Test)
- 💬 Chat history for the current session
- 🖥️ Clean Streamlit chat UI

## Architecture

```
USER
  ↓
STREAMLIT / CLI
  ↓
LANGCHAIN AGENT
  ↓
GOOGLE GEMINI
  ↓
TOOL DECISION
  ↓
TOOLS (calculator / code explainer / pytest / ruff / black /
       list_project_files / read_project_file / search_project /
       web_search / documentation_search /
       git_status / git_log / git_diff / git_branch /
       github_get_repository / github_get_issues / github_get_pull_requests)
  ↓
TOOL RESULT
  ↓
GOOGLE GEMINI
  ↓
FINAL RESPONSE
  ↓
STREAMLIT / CLI
  ↓
USER
```

Code generation, debugging, code review, refactoring, commit-message generation, and documentation generation are **not** separate tools — Gemini reasons about the code/diff/file contents directly, guided by response-format instructions in the system prompt. Only capabilities that need something outside the LLM (exact arithmetic, static code parsing, running an external program, or reaching an external service like Tavily/Git/GitHub) are implemented as tools.

## Requirements

- Python 3.10+ (developed and tested on Python 3.14)
- A Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

## Installation

```powershell
python -m venv venv
```

Activate the virtual environment (Windows):

```powershell
venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Then open `.env` and paste in your real key(s):

```
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest
TAVILY_API_KEY=
GITHUB_TOKEN=
GITHUB_REPO=
```

`GOOGLE_API_KEY` is required. `GEMINI_MODEL` is optional — it defaults to `gemini-flash-lite-latest` if you leave it out, but you can point it at any Gemini model your key has access to. Note that free-tier keys have a small daily quota per model (e.g. `gemini-3.6-flash` is capped at 20 requests/day on the free tier) — lite models like the default generally have a much higher free allowance.

Phase 4's variables are all optional:
- `TAVILY_API_KEY` — get a key from [app.tavily.com](https://app.tavily.com). Without it, `web_search`/`documentation_search` return a clear "not configured" message instead of failing unexpectedly.
- `GITHUB_TOKEN` — a fine-grained, **read-only** token from [github.com/settings/tokens](https://github.com/settings/tokens). Without it, the GitHub tools still work anonymously for public repositories, just with a lower rate limit and no access to private repos.
- `GITHUB_REPO` — a default `"owner/repo"` the GitHub tools use when you don't name one explicitly, e.g. `GITHUB_REPO=anthropics/claude-code`.

The `.env` file is listed in `.gitignore` and is never committed, printed, or shown in the UI.

## Run the Application

```powershell
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Example Questions

Try these in the chat (some are also available as one-click buttons in the sidebar):

1. `What is Python?` — direct answer, no tool
2. `Calculate 125 * 48` → `6000`
3. `Explain this Python code:` followed by a code snippet
4. `Write a Python function to check whether a number is prime.` — code generation
5. `Debug this code:` followed by broken code — debugging
6. `Review this code:` followed by a snippet — code review
7. `Refactor this code:` followed by a snippet — refactoring
8. `Generate pytest tests for this function:` followed by a function — test generation
9. `Run the tests.` — runs the project's Pytest suite
10. `Check this code with Ruff:` followed by a snippet — lints with Ruff
11. `Format this code with Black:` followed by a snippet — formats with Black
12. `What is this project?` — inspects the real project and explains it
13. `What does tools.py do?` — reads the real file and explains its contents
14. `Where is run_ruff implemented?` — searches the real project for it
15. `What is the current LangChain agent API?` — documentation search (requires `TAVILY_API_KEY`)
16. `Find the official documentation for Python pathlib.` — documentation search
17. `What files have changed in my project?` — `git_status`
18. `Show me my recent commits.` — `git_log`
19. `What is my current Git branch?` — `git_branch`
20. `What did I change in agent.py?` — `git_diff`
21. `Create a commit message from my current changes.` — reads the real `git_diff`, then writes a message (never runs `git commit`)
22. `Show me the open issues for octocat/hello-world.` — `github_get_issues`
23. `Show me the open pull requests for octocat/hello-world.` — `github_get_pull_requests`
24. `Generate documentation for tools.py.` — reads the real file, then writes documentation as its answer
25. `Where is Japan?` — out-of-scope, returns the exact scope-refusal message

## How the Agent Works

1. You type a message in the Streamlit chat input (or the CLI prompt), optionally after picking a Quick Action mode in the sidebar.
2. Streamlit/CLI passes the full conversation to `agent.py`.
3. LangChain's `create_agent` sends it to Gemini.
4. Gemini reads the message and decides: *"Do I need a tool for this, or can I answer/reason about it directly?"*
5. Direct-answer tasks (Q&A, code generation, debugging, review, refactoring, test generation, commit-message generation, documentation generation) are handled by Gemini's own reasoning, following the response format described in the system prompt — often *after* first calling a read-only tool (`git_diff`, `read_project_file`, etc.) to ground that reasoning in real data.
6. Tool tasks (math, structural code explanation, running Pytest/Ruff/Black, project inspection, web/documentation search, Git inspection, GitHub lookups) are dispatched to the matching function in `tools.py`.
7. Each tool returns a plain-text result — no tool ever calls the Gemini API itself.
8. That result is fed back into the conversation for Gemini to read. Web/documentation search results are explicitly labeled as untrusted external content in both the tool output and the system prompt, so Gemini treats them as reference information only — never as instructions.
9. Gemini writes the final, human-friendly answer using the tool's result.
10. Streamlit displays the answer, and — if a tool was used — a small "🔧 Using/Running X" box showing exactly what was passed in and what came back. The CLI prints the same information as `[TOOL CALL]`/`[TOOL INPUT]`/`[TOOL RESULT]` sections.

## Important Files

- **`app.py`** — the Streamlit UI only. Renders the sidebar (features, Quick Actions, example questions), chat history, input box, and tool-call boxes. Calls into `agent.py` for anything AI-related; contains no LLM or tool logic itself.
- **`agent.py`** — the agent itself. Loads the API key, configures the Gemini model, defines the system prompt (including the Phase 2 response-format and Phase 4 external-knowledge/Git/GitHub instructions), and builds the LangChain agent (`create_agent`) with all seventeen tools attached.
- **`tools.py`** — the seventeen tools, each a plain Python function wrapped in LangChain's `@tool` decorator:
  - `calculator` — parses the expression with Python's `ast` module and evaluates it node-by-node against an allow-list of operators. Never calls `eval()`.
  - `explain_python_code` — parses code with `ast` to list its structure **without running it**.
  - `run_pytest` — runs this project's own fixed `tests/` suite via `python -m pytest` and reports the exit code and output.
  - `run_ruff` — lints either pasted code (via stdin, nothing written to disk) or a project file (path validated against directory traversal).
  - `run_black` — formats pasted code (via stdin) or checks formatting of a project file and shows the diff **without ever overwriting it**.
  - `list_project_files` — returns this project's real file/folder structure as an indented tree, excluding noise folders (`venv`, `.git`, caches).
  - `read_project_file` — reads the real contents of a project source file (read-only), rejecting paths outside the project, `.env`, other credential-like files, binary files, and files over 1 MB.
  - `search_project` — searches this project's real source files for a function/class/variable/import/text and returns matching file paths and line numbers.
  - `web_search` — searches the web via Tavily; requires `TAVILY_API_KEY`, caps results to 5, and labels results as untrusted external content.
  - `documentation_search` — same as `web_search`, but biases toward official documentation domains for recognized frameworks/libraries.
  - `git_status` / `git_log` / `git_diff` / `git_branch` — read-only local Git inspection, sandboxed to this project's own repository (no parent-directory search), with capped commit/diff output and automatic redaction of any credential-like file's diff content.
  - `github_get_repository` / `github_get_issues` / `github_get_pull_requests` — read-only GitHub lookups via PyGithub; `GITHUB_TOKEN` is optional (falls back to anonymous, rate-limited access), and results are capped to 10 items.
- **`conftest.py`** — empty file at the project root so `pytest` can resolve `import tools` / `import agent` from `tests/` without a package layout.
- **`tests/`** — the project's own automated test suite (`test_tools.py`, `test_agent.py`, `test_logger.py`, `test_web_tools.py`, `test_git_tools.py`, `test_github_tools.py`), which is also what the `run_pytest` tool executes when you ask the assistant to "run the tests."

## Developer Tools (Phase 2)

| Capability | How it's implemented |
|---|---|
| Code generation | Gemini reasoning, structured as Explanation / Code / Usage |
| Debugging | Gemini reasoning, structured as Problem / Cause / Fixed Code / Explanation |
| Code review | Gemini reasoning, structured as Issues Found / Suggestions / Improved Code |
| Refactoring | Gemini reasoning, structured as Original Problem / Refactored Code / Improvements |
| Test generation | Gemini reasoning, output as pytest-style functions |
| Running tests | `run_pytest` tool — runs the project's fixed `tests/` folder only |
| Code quality | `run_ruff` tool — lints pasted code (stdin) or a project file |
| Formatting | `run_black` tool — formats pasted code (stdin) or checks a project file (read-only) |
| Project structure | `list_project_files` tool — real file/folder tree, noise folders excluded |
| Reading project files | `read_project_file` tool — real file contents, sandboxed to the project root |
| Searching the project | `search_project` tool — real file/line matches for a function/class/text query |

## Web Search, Documentation, Git & GitHub (Phase 4)

| Capability | How it's implemented |
|---|---|
| Web search | `web_search` tool — Tavily search, capped at 5 results, labeled as untrusted content |
| Documentation search | `documentation_search` tool — Tavily search biased toward official doc domains for recognized topics |
| Current API/framework info | Gemini is instructed to verify with `documentation_search`/`web_search` rather than presenting old training knowledge as current, and to say so if it can't verify |
| Git status | `git_status` tool — staged/modified/deleted/untracked files, read-only |
| Git log | `git_log` tool — recent commits, capped at 20 |
| Git diff | `git_diff` tool — uncommitted changes (working tree, falls back to staged), summarized instead of shown in full past 6000 characters, credential-like files redacted |
| Git branch | `git_branch` tool — current branch + local branches, read-only |
| GitHub repository info | `github_get_repository` tool — description, stars, default branch, open issue count |
| GitHub issues | `github_get_issues` tool — up to 10 real issues (pull requests filtered out) |
| GitHub pull requests | `github_get_pull_requests` tool — up to 10 real pull requests |
| Commit message generation | Gemini reasoning over the real `git_diff` output — never runs `git commit` |
| Documentation generation | Gemini reasoning over real `read_project_file`/`search_project` output — produces text only, never writes files |

## Core Concepts

- **Google Gemini** — the LLM. It understands language and generates responses.
- **A Tool** — a specific capability the agent is allowed to call.
- **An Agent** — uses the LLM to decide *what action to take*: answer/reason directly, or call a tool first.
- **LangChain** — the framework that connects the LLM, the tools, and the decision loop together (`create_agent`).
- **Streamlit** — the web UI the user actually interacts with.

## A Note on LangChain Versions

This project uses **LangChain 1.x**, which replaced the older `initialize_agent` / `AgentExecutor` pattern from pre-1.0 tutorials with a single `create_agent()` function (built on LangGraph internally). If you've seen older LangChain tutorials using `AgentExecutor` or `initialize_agent`, that API is no longer used here — `create_agent` is the current recommended way to build a tool-using agent.

## Security Notes

- The API key is only ever read from `.env` via `python-dotenv` — never hard-coded, printed, or displayed.
- The calculator never uses `eval()` on raw text; it validates a parsed syntax tree against an operator allow-list.
- The code explanation tool never executes user-submitted code; it only inspects the syntax tree.
- **Arbitrary code execution and unrestricted shell commands are intentionally not supported.** `run_pytest`, `run_ruff`, and `run_black` are the only tools that shell out to anything, and each is deliberately restricted:
  - Every subprocess call uses an explicit argument list (never `shell=True`), so there is no shell/command injection.
  - `run_pytest` always runs this project's fixed `tests/` folder — it takes no arguments and cannot be pointed anywhere else.
  - Any `file_path` passed to `run_ruff`/`run_black` is resolved and checked against the project root; anything that would escape it (`..`, an absolute path elsewhere) is rejected before it touches the filesystem.
  - Pasted `code` for `run_ruff`/`run_black` is passed over stdin and is never written to disk.
  - `run_black` never overwrites a project file — on a file it only reports whether it's formatted and shows the diff.
- **`list_project_files`, `read_project_file`, and `search_project` are read-only and sandboxed to the project root**, the same way `run_ruff`/`run_black` are:
  - Any path is resolved and checked against the project root; anything that would escape it (`..`, another drive, an absolute path elsewhere) is rejected.
  - The `.env` file and other credential-like files/extensions (`.pem`, `.key`, names containing "secret"/"credential"/"password", etc.) can never be read or searched, even by exact name.
  - Noise/third-party folders (`venv`, `.git`, `__pycache__`, tool caches) are excluded from listings, reads, and search results.
  - `read_project_file` rejects binary files and files over 1 MB with a clear error instead of trying to dump them.
  - None of the three can write, execute, or delete anything.
- **`web_search`/`documentation_search` never leak `TAVILY_API_KEY`**: it's read only from `.env`, never printed, logged, or included in a tool result, and a missing key produces a clear error instead of a crash.
  - Search results are external, untrusted web content. Both the tool output and the system prompt label them as reference information only — Gemini is explicitly instructed to never follow instructions found inside a search result (prompt-injection defense), and never let one override the system prompt, scope, or security rules.
  - Result count (5) and per-result snippet length (500 characters) are capped so a single search can never flood the model's context.
- **`git_status`/`git_log`/`git_diff`/`git_branch` are strictly read-only** — there is no generic "run git command" tool, and nothing here can stage, commit, push, reset, clean, or switch/delete branches.
  - The repository is only opened at this project's own root (no parent-directory search), so these tools can never reach an unrelated repository elsewhere on disk.
  - `git_log` caps commit count at 20; `git_diff` summarizes (instead of showing in full) any diff over 6000 characters, and redacts the content of any credential-like file's diff hunk even if it were accidentally tracked.
  - A missing `git` executable or a non-Git directory produces a clear error instead of crashing the whole app (all tool imports stay resilient to Git not being installed).
- **`github_get_repository`/`github_get_issues`/`github_get_pull_requests` are read-only and least-privilege** — nothing here can create, close, comment on, or modify issues/PRs, or change repository settings.
  - `GITHUB_TOKEN` is optional and never printed, logged, or included in a tool result; without it, the tools fall back to anonymous, rate-limited access for public repositories.
  - Issue/PR counts are capped at 10, and results are only ever the real data PyGithub returned — never invented.
- Errors are shown to the user as short, friendly messages — full details are only printed to the terminal for debugging.
- `logger.py`'s secret redaction also recognizes GitHub token formats (`ghp_`/`gho_`/`github_pat_`/etc.) and Tavily key formats (`tvly-`), in addition to the existing Google API key and generic `KEY=`/`TOKEN=`/`SECRET=` patterns.

## Testing

Run the project's own test suite:

```powershell
python -m pytest tests -v
```

Or ask the assistant directly in the chat: *"Run the tests."*

Tests mock the LangChain agent object where relevant (`ask_agent`'s tool-call extraction) so the suite never depends on a live Gemini API call or a configured API key. `run_pytest`/`run_ruff`/`run_black` are tested against the real installed executables since they're safe, read-only/stdin-only operations. The Phase 4 tools are tested the same way `ask_agent` is — by mocking at the external-service boundary (`tools.TavilyClient`, `tools._get_repo`, `tools._get_github_client`) so the suite never makes a real network call, never needs a real `TAVILY_API_KEY`/`GITHUB_TOKEN`, and never needs a real `git` executable installed.

## Version 4 Ideas (not implemented)

These are intentionally left out to keep this project simple and easy to learn from — Phase 4 is about external knowledge and Git/GitHub *assistance*, not autonomous execution:

- RAG (retrieval-augmented generation) / vector databases
- Multi-agent architectures / LangGraph
- Sandboxed/unrestricted code execution
- Automatic test/fix/re-run loops
- Automatic Git operations (commit, push, branch create/delete, reset, clean)
- Automatic GitHub operations (creating/closing/commenting on issues or PRs, merging)
- File upload / analysis
- Persistent long-term memory / database
- Docker / cloud deployment
- User accounts / authentication
