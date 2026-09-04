"""
Tests for tools.py.

These exercise the tools directly (via .invoke(...), the same way LangChain
calls them) without needing the Gemini API - the calculator and code
explainer are pure Python, and run_ruff/run_black only shell out to the
already-installed ruff/black executables.
"""

from tools import (
    PROJECT_ROOT,
    _build_tree,
    calculator,
    explain_python_code,
    list_project_files,
    read_project_file,
    run_black,
    run_ruff,
    search_project,
)


def test_calculator_multiplication():
    assert calculator.invoke({"expression": "125 * 48"}) == "6000"


def test_calculator_parentheses_and_precedence():
    assert calculator.invoke({"expression": "(20 + 5) * 4"}) == "100"


def test_calculator_power():
    assert calculator.invoke({"expression": "2 ** 10"}) == "1024"


def test_calculator_division_by_zero():
    result = calculator.invoke({"expression": "10 / 0"})
    assert "division by zero" in result.lower()


def test_calculator_rejects_unsafe_input():
    result = calculator.invoke({"expression": "__import__('os')"})
    assert "error" in result.lower()


def test_calculator_empty_expression():
    result = calculator.invoke({"expression": "   "})
    assert "error" in result.lower()


def test_explain_python_code_finds_structure():
    code = "for i in range(5):\n    print(i)"
    result = explain_python_code.invoke({"code": code})
    assert "For-loop" in result
    assert "range(5)" in result


def test_explain_python_code_reports_syntax_error():
    result = explain_python_code.invoke({"code": "def broken(:"})
    assert "syntax error" in result.lower()


def test_run_ruff_flags_unused_import():
    result = run_ruff.invoke({"code": "import os\n\nprint('hi')\n"})
    assert "os" in result


def test_run_ruff_clean_code_reports_no_issues():
    result = run_ruff.invoke({"code": "print('hi')\n"})
    assert "no issues" in result.lower()


def test_run_ruff_requires_exactly_one_argument():
    assert "error" in run_ruff.invoke({}).lower()
    assert (
        "error" in run_ruff.invoke({"code": "x = 1", "file_path": "tools.py"}).lower()
    )


def test_run_ruff_accepts_project_root_directory():
    result = run_ruff.invoke({"file_path": "."})
    assert "outside the project directory" not in result
    assert "was not found" not in result.lower()


def test_run_black_formats_pasted_code():
    messy = "def add(a,b):\n return a+b\n"
    formatted = run_black.invoke({"code": messy})
    assert formatted == "def add(a, b):\n    return a + b\n"


def test_run_black_checks_real_project_file():
    result = run_black.invoke({"file_path": "conftest.py"})
    assert "outside the project directory" not in result
    assert "was not found" not in result.lower()
    assert "conftest.py" in result


def test_run_black_accepts_project_root_directory():
    result = run_black.invoke({"file_path": "."})
    assert not result.startswith("Error")
    assert "the project" in result.lower()


def test_run_black_accepts_project_root_with_trailing_slash():
    result = run_black.invoke({"file_path": "./"})
    assert not result.startswith("Error")
    assert "the project" in result.lower()


def test_run_black_file_path_rejects_directory_traversal():
    result = run_black.invoke({"file_path": "../outside.py"})
    assert "outside the project directory" in result


def test_run_black_file_path_rejects_path_traversal_deep():
    result = run_black.invoke({"file_path": "../../something"})
    assert "outside the project directory" in result


def test_run_black_file_path_rejects_absolute_path_outside_project():
    result = run_black.invoke(
        {"file_path": "../../../../Windows/System32/drivers/etc/hosts"}
    )
    assert "outside the project directory" in result


def test_run_black_file_path_rejects_dot_env():
    result = run_black.invoke({"file_path": ".env"})
    assert "cannot be checked" in result.lower()
    assert "GOOGLE_API_KEY" not in result


def test_run_black_file_path_missing_file():
    result = run_black.invoke({"file_path": "does_not_exist.py"})
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# list_project_files
# ---------------------------------------------------------------------------


def test_list_project_files_shows_real_project_structure():
    result = list_project_files.invoke({})
    assert "agent.py" in result
    assert "tools.py" in result
    assert "tests" in result


def test_list_project_files_excludes_noise_folders():
    result = list_project_files.invoke({})
    assert "venv/" not in result
    assert "__pycache__" not in result
    assert ".git/" not in result


def test_build_tree_handles_missing_directory_safely():
    lines: list[str] = []
    _build_tree(PROJECT_ROOT / "does_not_exist_dir", lines, prefix="")
    assert lines == []


# ---------------------------------------------------------------------------
# read_project_file
# ---------------------------------------------------------------------------


def test_read_project_file_reads_real_file():
    result = read_project_file.invoke({"file_path": "conftest.py"})
    assert "Contents of 'conftest.py'" in result
    assert "pytest" in result.lower()


def test_read_project_file_rejects_outside_project_root():
    result = read_project_file.invoke({"file_path": "../outside.py"})
    assert "outside the project directory" in result


def test_read_project_file_rejects_path_traversal():
    result = read_project_file.invoke(
        {"file_path": "../../Windows/System32/drivers/etc/hosts"}
    )
    assert "outside the project directory" in result


def test_read_project_file_rejects_dot_env():
    result = read_project_file.invoke({"file_path": ".env"})
    assert "cannot be read" in result.lower()
    assert "GOOGLE_API_KEY" not in result


def test_read_project_file_rejects_venv_files():
    result = read_project_file.invoke({"file_path": "venv/pyvenv.cfg"})
    assert "cannot be read" in result.lower()


def test_read_project_file_missing_file_handled_safely():
    result = read_project_file.invoke({"file_path": "does_not_exist.py"})
    assert "not found" in result.lower()


def test_read_project_file_empty_path():
    result = read_project_file.invoke({"file_path": "   "})
    assert "error" in result.lower()


def test_read_project_file_rejects_binary_file():
    scratch = PROJECT_ROOT / "_scratch_binary_test.bin"
    scratch.write_bytes(b"\x00\x01\x02\x03binarydata")
    try:
        result = read_project_file.invoke({"file_path": scratch.name})
        assert "cannot be inspected as text" in result.lower()
    finally:
        scratch.unlink()


def test_read_project_file_rejects_oversized_file():
    scratch = PROJECT_ROOT / "_scratch_oversized_test.txt"
    scratch.write_text("x" * 1_200_000)
    try:
        result = read_project_file.invoke({"file_path": scratch.name})
        assert "too large to inspect safely" in result.lower()
    finally:
        scratch.unlink()


# ---------------------------------------------------------------------------
# search_project
# ---------------------------------------------------------------------------


def test_search_project_finds_real_function_definition():
    result = search_project.invoke({"query": "def run_ruff"})
    assert "tools.py" in result
    assert "def run_ruff" in result


def test_search_project_no_matches():
    needle = "zzz_no_such_" + "token_in_this_project_zzz"
    result = search_project.invoke({"query": needle})
    assert "no matches found" in result.lower()


def test_search_project_empty_query():
    result = search_project.invoke({"query": ""})
    assert "error" in result.lower()


def test_search_project_skips_dot_env():
    result = search_project.invoke({"query": "GOOGLE_API_KEY"})
    assert ".env:" not in result
