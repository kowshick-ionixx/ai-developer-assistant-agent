# 🤖 AI Developer Assistant

## Overview

A beginner-friendly AI agent built with **Python**, **LangChain**, **Google Gemini**, and **Streamlit** that behaves like a junior software developer. It answers programming questions, does math, explains code, and — as of Phase 2 — generates, debugs, reviews, refactors, and tests code, and can run Pytest/Ruff/Black on request. It decides for itself, on every message, whether it needs a tool or can just answer directly. It is also **project-aware**: it can inspect this project's own real structure, files, and source code (via `list_project_files`/`read_project_file`/`search_project`) to answer questions about itself accurately, instead of guessing. As of Phase 4, it can also reach outside the project: **web/documentation search** (via Tavily) for current framework/API information, **read-only Git inspection**, **read-only GitHub repository/issue/PR lookups**, and **commit-message and documentation generation** grounded in the project's real diff/files. As of Phase 5, it can also check Python files for syntax errors (`check_python_syntax`), analyze real pytest failures and tracebacks by combining `run_pytest`'s actual output with `read_project_file`/`search_project`, and verify a fix by rerunning the suite — always reporting real results, never fabricated ones. It also supports 🎤 voice questions and 📎 document upload (code, text, PDF, DOCX) directly from the chat bar. As of Phase 6, it can carry out **multi-step development tasks** — plan, inspect, propose a file change, and (only once a human approves it in the sidebar) apply it and re-test — instead of only answering one question at a time.

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

**Phase 5**
- ✅ Controlled test execution — `run_pytest` reports the real exit code plus captured stdout/stderr, bounded by a subprocess timeout
- 🩺 Syntax checking (`check_python_syntax`) — parses (never executes) one file, a folder, or the whole project to find real `SyntaxError`s
- 🧵 Traceback analysis — explains Problem / Cause / Location / Suggested fix using only the file and line actually present in a traceback
- 🧪 Test failure analysis — combines `run_pytest`'s real output with `read_project_file`/`search_project` to explain why a specific test actually failed
- 🔁 Fix-and-verify workflow — since there is no file-editing tool, the assistant explains the needed code change and only claims an issue "is fixed" after rerunning `run_pytest` and seeing the real result
- 📊 Regression checking — reruns the full suite after a fix and reports actual before/after pass/fail counts

**Voice & Document Upload**
- 🎤 Voice questions — record from the chat bar's built-in microphone button; transcribed to text by Gemini's native audio understanding (no separate speech-to-text service)
- 📎 Document upload — attach `.py`/`.txt`/`.md`/`.json`/`.csv`/`.yaml`/`.yml`/`.toml`/`.xml`/`.html`/`.css`/`.js`/`.ts`/`.pdf`/`.docx` files directly from the chat bar; content is validated, size-limited, secret-redacted, and added as labeled, untrusted reference context
- 🔊 Optional "Read aloud" button on any answer, using the browser's own text-to-speech

**Phase 6**
- 📋 Task planning — a short, concrete numbered plan for a multi-step development request
- 🔍 Autonomous tool selection across a whole conversation (inspect → search docs → propose → test)
- 📝 Controlled file creation/modification (`propose_file_change`) — proposes a complete new file's contents; never writes anything by itself
- ✅ Human approval gate (`apply_approved_change`) — a change can only be written to disk after you click Approve in the sidebar; the tool itself refuses an unapproved change_id, even if asked to skip approval
- 📄 PDF/document requirements → project generation — reuses the existing document-upload feature as the requirements source for the same propose → approve → apply → test flow
- 🔁 Test → analyze → fix → retest loop, guided by a stated maximum of 3 repair attempts per task
- 🔧 Workflow-state observability (`[WORKFLOW]`/`[APPROVAL]` log sections, and a "Workflow: ..." caption in the UI) derived from which tools actually ran

**All phases**
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
TOOLS (calculator / code explainer / pytest / ruff / black / check_python_syntax /
       list_project_files / read_project_file / search_project /
       web_search / documentation_search /
       git_status / git_log / git_diff / git_branch /
       github_get_repository / github_get_issues / github_get_pull_requests /
       propose_file_change / apply_approved_change)
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
25. `Check my project for syntax errors.` — `check_python_syntax`
26. `Run my tests and explain why any tests are failing.` — `run_pytest`, then reads the failing test/source to explain the real cause
27. `Where is Japan?` — out-of-scope, returns the exact scope-refusal message

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

- **`app.py`** — the Streamlit UI only. Renders the sidebar (features, Quick Actions, attached files, Phase 6 pending approvals, the AI Features reference panel), chat history, input box (with built-in 📎 attach and 🎤 record buttons), and tool-call boxes. Calls into `agent.py`/`documents.py`/`workflow.py` for anything AI-related; contains no LLM or tool logic itself.
- **`agent.py`** — the agent itself. Loads the API key, configures the Gemini model, defines the system prompt (including the Phase 2 response-format, Phase 4 external-knowledge/Git/GitHub instructions, Phase 5 execution/error-analysis/security instructions, and the Phase 6 controlled-development-task instructions), and builds the LangChain agent (`create_agent`) with all twenty tools attached. Also exposes `transcribe_audio()` (voice → text via Gemini's native audio understanding) and `create_plan()` (Phase 6 task planner) as separate, single-completion calls that never build a second tool-using agent.
- **`documents.py`** — validates, parses, and sanitizes documents attached via the chat bar (text formats directly, `.pdf` via `pypdf`, `.docx` via `python-docx`), redacting secret-looking content before it is ever shown or sent to the model.
- **`workflow.py`** — Phase 6's plain data structures: `WorkflowStatus`/`WorkflowState` (concise, observable task progress with controlled state transitions) and the pending-change registry (`register_change`/`approve_change`/`reject_change`/`list_pending_changes`) behind the human-approval gate. Holds no filesystem or LangChain logic of its own.
- **`tools.py`** — twenty tools, each a plain Python function wrapped in LangChain's `@tool` decorator:
  - `calculator` — parses the expression with Python's `ast` module and evaluates it node-by-node against an allow-list of operators. Never calls `eval()`.
  - `explain_python_code` — parses code with `ast` to list its structure **without running it**.
  - `run_pytest` — runs this project's own fixed `tests/` suite via `python -m pytest` and reports the exit code and output.
  - `run_ruff` — lints either pasted code (via stdin, nothing written to disk) or a project file (path validated against directory traversal).
  - `run_black` — formats pasted code (via stdin) or checks formatting of a project file and shows the diff **without ever overwriting it**.
  - `check_python_syntax` — parses one file, a folder, or the whole project with `ast` to report real `SyntaxError`s, **without ever executing anything**.
  - `list_project_files` — returns this project's real file/folder structure as an indented tree, excluding noise folders (`venv`, `.git`, caches).
  - `read_project_file` — reads the real contents of a project source file (read-only), rejecting paths outside the project, `.env`, other credential-like files, binary files, and files over 1 MB.
  - `search_project` — searches this project's real source files for a function/class/variable/import/text and returns matching file paths and line numbers.
  - `web_search` — searches the web via Tavily; requires `TAVILY_API_KEY`, caps results to 5, and labels results as untrusted external content.
  - `documentation_search` — same as `web_search`, but biases toward official documentation domains for recognized frameworks/libraries.
  - `git_status` / `git_log` / `git_diff` / `git_branch` — read-only local Git inspection, sandboxed to this project's own repository (no parent-directory search), with capped commit/diff output and automatic redaction of any credential-like file's diff content.
  - `github_get_repository` / `github_get_issues` / `github_get_pull_requests` — read-only GitHub lookups via PyGithub; `GITHUB_TOKEN` is optional (falls back to anonymous, rate-limited access), and results are capped to 10 items.
  - `propose_file_change` (Phase 6) — registers a pending file create/modify in `workflow.py`; validated against the same project-root/excluded/blocked-file rules as `read_project_file`, but **never writes anything**.
  - `apply_approved_change` (Phase 6) — writes a previously proposed change to disk, but only if `workflow.approve_change()` has already been called for that `change_id` — which only the Streamlit UI's Approve button (or the CLI's approval prompt) can do. The agent has no tool that can approve its own change.
- **`conftest.py`** — empty file at the project root so `pytest` can resolve `import tools` / `import agent` from `tests/` without a package layout.
- **`tests/`** — the project's own automated test suite (`test_tools.py`, `test_agent.py`, `test_logger.py`, `test_web_tools.py`, `test_git_tools.py`, `test_github_tools.py`, `test_documents.py`, `test_workflow.py`, `test_dev_tools.py`, `test_app.py`), which is also what the `run_pytest` tool executes when you ask the assistant to "run the tests."

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
| Syntax checking | `check_python_syntax` tool — parses (never executes) file(s) to find real syntax errors |

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

## Code Execution, Testing & Error Analysis (Phase 5)

| Capability | How it's implemented |
|---|---|
| Controlled test execution | `run_pytest` tool — real exit code + captured stdout/stderr, bounded by `_SUBPROCESS_TIMEOUT_SECONDS` |
| Syntax checking | `check_python_syntax` tool — parses file(s) with `ast`, never executes them |
| Traceback analysis | Gemini reasoning over the actual traceback text — Problem / Cause / Location / Suggested fix, never inventing a file or line not present in it |
| Test failure analysis | Gemini reasoning over `run_pytest`'s real output plus `read_project_file`/`search_project` on the failing test and the source it exercises |
| Fix-and-verify workflow | Gemini explains the needed code change and (Phase 6) can propose it as a file change for approval; only reruns `run_pytest` and only claims "fixed" after the real rerun confirms an *applied* change |
| Regression testing | Rerunning `run_pytest` after a fix and comparing the two real before/after pass/fail counts |
| Security | No generic command-execution tool exists at all — arbitrary PowerShell/CMD/shell commands, unrestricted Python execution, and destructive/file-modifying requests are refused by the system prompt rather than attempted |

## Phase 6 — Autonomous Software Development

Phase 6 turns multi-step development requests ("add a feature", "fix this bug", "generate this project from the uploaded PDF") into a controlled, human-approved sequence instead of a single Q&A answer — while reusing every existing tool and adding three new ones.

| Capability | How it's implemented |
|---|---|
| Request classification | Two layers: a "## Request Classification" section in the system prompt distinguishes a pure calculation ("Calculate 5 factorial" → `calculator`) from a development request that merely mentions a math concept ("Add a function to calculate the factorial of a number" → propose real code); and `agent.classify_request()` — a single, dedicated Gemini completion call (DEVELOPMENT/OTHER) used by `run_agent_turn()` (below) to decide whether a turn needs the auto-continuation loop at all, so calculations/Q&A never pay for or trigger it. `calculator`'s own docstring was also hardened with an explicit exclusion and example |
| Task planning | `agent.create_plan()` — a single, separate Gemini completion call (like `transcribe_audio`) that returns a short numbered plan preview; the live `[WORKFLOW] State: PLANNING` marker is logged by `run_agent_turn()` at the start of every classified development turn, ahead of the main agent's own inline numbered plan (required by the Phase 6 system-prompt section) |
| Reliable end-to-end continuation | `agent.run_agent_turn()` — wraps `ask_agent()` in a bounded auto-continuation loop (`MAX_AUTO_CONTINUE_STEPS = 5`). **Fixes the reported failure mode**: `create_agent`'s tool-calling loop only keeps going while the model keeps requesting tools, and live testing confirmed the model sometimes stops after stating its plan/findings as plain text with zero tool calls, silently ending the turn at INSPECTING/PLANNING. `run_agent_turn()` detects a turn that made no progress toward a real pause point (a change actually awaiting approval) and isn't a question back to the user, and nudges the *same* agent (a plain follow-up message, never a second agent) to keep acting — it bails out the moment two consecutive attempts produce no tool call at all, the moment a full test run passes with nothing pending, or after 5 attempts, so it can never loop forever (same principle as `MAX_REPAIR_ATTEMPTS`). Only applied to requests `classify_request()` marks as development — a calculation or direct Q&A is passed straight to `ask_agent()` unchanged |
| Autonomous tool selection | The existing LangChain `create_agent` tool-calling loop, guided by a new "Phase 6" section in the system prompt — the same mechanism Phases 1-5 already use, just with more tools available in one conversation, now kept moving turn-to-turn by `run_agent_turn()` above |
| Workflow state | `workflow.WorkflowStatus`/`WorkflowState` — an enum (`PLANNING`, `INSPECTING`, `SEARCHING_DOCUMENTATION`, `PROPOSING_CHANGE`, `WAITING_FOR_APPROVAL`, `IMPLEMENTING`, `TESTING`, `ANALYZING`, `FIXING`, `RETESTING`, `REGRESSION_TESTING`, `REVIEWING`, `COMPLETED`, `FAILED`) with controlled transitions (`transition_to()` raises on an invalid jump) and a dataclass holding only concise summaries (plan, completed/pending steps, file list, retry count) — never chain-of-thought or secrets. `agent._derive_workflow_states()` infers the finer-grained states (e.g. `PROPOSING_CHANGE` vs `IMPLEMENTING`, `RETESTING` vs `REGRESSION_TESTING`, an `ANALYZING`→`FIXING` pair before a fix's proposal) from the real sequence of tool calls and `run_pytest`'s own real "Exit code:" output in a turn — never fabricated |
| Controlled file creation/modification | `propose_file_change` — validates the path (same rules as `read_project_file`) and registers a `ProposedChange`; **never writes to disk**. A "modify" proposal that is drastically shorter than the file it replaces (e.g. the model truncating a large file with a placeholder comment instead of reproducing it in full — observed once in live testing) is flagged `risk="high"` with an explicit warning, since a human approver reviewing a large diff might not otherwise notice a silent truncation |
| Human approval | `workflow.approve_change()`/`reject_change()`, called only from the Streamlit sidebar's "🔧 Pending Approvals" section (Approve/Reject buttons) or an equivalent CLI prompt — never from a tool the agent itself can call |
| Applying a change | `apply_approved_change` — refuses to write anything unless `ProposedChange.approved` is already `True`; re-validates path safety independently at apply time too |
| PDF/document → project generation | Reuses the existing document-upload feature (Part of Phase 5.5): the uploaded document's extracted, sanitized text becomes the "ATTACHED DOCUMENT CONTEXT" the agent treats as requirements, then follows the same propose → approve → apply → test flow |
| Test → analyze → fix → retest loop | `run_pytest` after an applied change, `read_project_file`/`search_project` plus the real failure output to diagnose, another `propose_file_change` for the fix. **The `MAX_REPAIR_ATTEMPTS = 3` limit is enforced in code, not just prompted**: `workflow.py` tallies applies per file and clears the tally only on a passing `run_pytest`; once a file has been applied 3 times with no passing run in between, `apply_approved_change` itself refuses to write it again (with the real reason in its output) until a human resets the counter (a "🔄 Reset repair counter" control appears in the sidebar once any file hits the limit) |
| Cross-turn change lookup | `list_pending_changes` — a read-only tool that lists every real pending change (id, file, approved status, risk); the system prompt tells the model to call it before proposing a duplicate change or assuming an earlier turn's change_id/approval status is still accurate, instead of relying on its own memory of the conversation |
| Regression testing | Rerunning the full `run_pytest` suite after a change and reporting the actual before/after pass/fail counts (same mechanism as Phase 5) |
| Git-aware development | `git_status` before starting, `git_diff` after applying — still strictly read-only; there is still no tool that can commit, push, or otherwise change the repository |
| Observability | New `[PHASE 6]`/`[PLANNER]`/`[WORKFLOW]`/`[APPROVAL]` log sections in `logger.py`, and a "🔧 Workflow: ..." caption in the Streamlit UI, derived from which tools actually ran this turn |
| Security | Everything above reuses the exact same project-root/excluded/blocked-file checks as the read-only tools; the human-approval gate is enforced in code (`ProposedChange.approved`), not just by a prompt instruction — see [Security Notes](#security-notes) |

**Honest scope note:** the workflow-status trail shown in the UI/logs is derived *after* a turn's tool calls already ran (LangChain's `create_agent` loop is synchronous), not pushed live per call — it is accurate, observable progress reporting, not a real-time stream; fixing this for real would mean switching `ask_agent()` from `agent.invoke()` to `agent.stream()` and reworking how tests stand in for the agent, which was judged too large a change to make as a side effect of an audit and is left as documented, known behavior rather than "fixed" without real verification. The `MAX_REPAIR_ATTEMPTS = 3` limit and cross-turn change lookup, by contrast, **are now real, code-level fixes** (see the table above) rather than prompt-only policy: `apply_approved_change` itself refuses a file's 4th consecutive apply without a passing test in between, and `list_pending_changes` gives the model (and a human) a ground-truth read of pending changes instead of relying on conversation memory. Neither eliminates every possible model mistake on its own - a model can still occasionally re-propose a duplicate change instead of calling `list_pending_changes` first - but the structural gates underneath (the approval flag, the per-file repair tally) hold regardless of what the model does, which is the actual backstop. Both the request-classification fix and the truncation-detection safeguard were added after a live audit turned up the exact scenarios they now catch — a compound "add X, create tests, run tests, verify" request momentarily bypassing `propose_file_change` entirely, and a large "modify" proposal that silently dropped most of the target file's real content.

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
- **There is no generic "run a command" tool, and Phase 5 does not add one.** `run_pytest`, `run_ruff`, `run_black`, and `check_python_syntax` remain the only tools that touch the filesystem/an external process, each restricted exactly as described above; `check_python_syntax` only ever calls `ast.parse()` on file contents it already validated with the same path-safety/exclusion/blocked-file checks as `read_project_file` — it never executes anything.
- Every subprocess-based tool (`run_pytest`/`run_ruff`/`run_black`) enforces a fixed timeout, so a hung process can never block the agent indefinitely; a timeout is always reported honestly ("took too long") instead of as a fabricated success.
- The system prompt explicitly instructs the assistant to refuse requests for arbitrary PowerShell/CMD/shell commands and unrestricted Python execution, explaining that no such tool exists rather than attempting the request another way. It may name environment variables (`GOOGLE_API_KEY`, `TAVILY_API_KEY`, `GITHUB_TOKEN`) but is instructed to never state or guess their values.
- **File creation/modification (Phase 6) is controlled and human-approved, never blind.** `propose_file_change` validates the path (same project-root/excluded/blocked-file rules as `read_project_file`) and only ever registers a pending change — it cannot write to disk. `apply_approved_change` re-validates path safety independently and refuses to write anything unless `workflow.approve_change()` has already been called for that exact `change_id` — and that function is only ever called from the human-facing Streamlit sidebar (Approve/Reject buttons) or CLI approval prompt, never from a tool the agent itself can call. Asking the assistant to "skip approval" or "apply it directly" does not bypass this: the system prompt refuses, and even if it didn't, the tool itself would still refuse. There is still no tool that can delete a file.

## Testing

Run the project's own test suite:

```powershell
python -m pytest tests -v
```

Or ask the assistant directly in the chat: *"Run the tests."*

Tests mock the LangChain agent object where relevant (`ask_agent`'s tool-call extraction) so the suite never depends on a live Gemini API call or a configured API key. `run_ruff`/`run_black`/`check_python_syntax` are tested against real inputs since they're safe, read-only/stdin-only/parse-only operations. `run_pytest` (`tests/test_execution_tools.py`) is tested by mocking `subprocess.run` instead of actually invoking it, since a real call would recursively re-run this whole suite as a subprocess — the mock still verifies real stdout/stderr capture, exit-code handling, the configured timeout, and error handling (missing executable, missing `tests/` folder). The Phase 4 tools are tested the same way `ask_agent` is — by mocking at the external-service boundary (`tools.TavilyClient`, `tools._get_repo`, `tools._get_github_client`) so the suite never makes a real network call, never needs a real `TAVILY_API_KEY`/`GITHUB_TOKEN`, and never needs a real `git` executable installed.

`documents.py` (`tests/test_documents.py`) is tested against real parsers (`pypdf`, `python-docx`) with in-memory files built and torn down inside the test, plus the same secret-redaction assertions used elsewhere. `workflow.py` (`tests/test_workflow.py`) and the Phase 6 tools (`tests/test_dev_tools.py`) are tested directly — including that `apply_approved_change` genuinely refuses to write an unapproved change and that approving one `change_id` never approves another. `tests/test_app.py` drives the real `app.py` script headlessly via Streamlit's own `AppTest` (with `run_agent_turn`/`transcribe_audio` mocked), so the attachment/voice/Phase 6 approval UI wiring is exercised end-to-end without a browser or a live API call. `run_agent_turn()`'s own auto-continuation/classification logic (`tests/test_agent.py`) is tested with `ask_agent`/`classify_request` mocked at the module level, so the nudge loop, the "never loop forever" bail-outs, and the completed/failed derivation are all verified without a live Gemini call.

## Ideas Not Implemented

These are intentionally left out to keep this project's autonomy controlled and easy to reason about:

- Sandboxed/unrestricted code execution or a generic "run any command" tool
- Fully autonomous, unattended file changes that skip human approval
- An interrupt that can force-stop a tool call already in flight mid-turn (the repair-attempt limit is now enforced in code, but only between tool calls - `apply_approved_change` checks the tally before it writes, not while some other, unrelated call is still running)
- A live, per-call streaming workflow-status feed (today's `[WORKFLOW]` trail is derived from a turn's tool calls after `agent.invoke()` already completed - accurate, but not real-time; see the Phase 6 honest-scope note above)
- Fully reliable cross-turn reuse of a specific proposed change_id (the model has `list_pending_changes` to check the ground truth now, but can still occasionally re-propose a fresh change instead of calling it first - the approval gate and the repair-attempt tally both still hold either way)
- Deleting files, or any destructive filesystem operation
- RAG (retrieval-augmented generation) / vector databases — document context uses direct inclusion with truncation, not chunking/embeddings
- Multi-agent architectures / a custom LangGraph orchestrator (Phase 6 reuses the existing single-agent `create_agent` tool-calling loop)
- Automatic Git operations (commit, push, branch create/delete, reset, clean)
- Automatic GitHub operations (creating/closing/commenting on issues or PRs, merging)
- Automatic deployment
- Persistent long-term memory / database (the Phase 6 pending-change registry is in-memory and process-local, not durable storage)
- Docker / cloud deployment
- User accounts / authentication / multi-user support (this app assumes one local, single-user session)
