"""
Tests for the Phase 5 controlled execution / error-analysis tools in tools.py.

run_pytest is mocked at the subprocess.run() boundary (the same pattern
test_git_tools.py/test_web_tools.py use for their external boundaries) since
actually invoking it here would recursively re-run this entire test suite as
a subprocess. check_python_syntax is exercised for real - it never shells
out or executes anything, it only parses source text with `ast`.
"""

import subprocess

import tools
from tools import PROJECT_ROOT, check_python_syntax, run_pytest


class _FakeCompletedProcess:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# ---------------------------------------------------------------------------
# run_pytest - stdout/stderr capture, exit code, timeout, error handling
# ---------------------------------------------------------------------------


def test_run_pytest_reports_actual_passed_summary(monkeypatch):
    def fake_run(*args, **kwargs):
        return _FakeCompletedProcess(stdout="3 passed in 0.10s\n", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_pytest.invoke({})
    assert "Exit code: 0" in result
    assert "3 passed in 0.10s" in result


def test_run_pytest_reports_actual_failure_summary(monkeypatch):
    def fake_run(*args, **kwargs):
        return _FakeCompletedProcess(
            stdout=(
                "FAILED tests/test_math.py::test_add - assert 2 == 8\n"
                "1 failed, 2 passed in 0.12s\n"
            ),
            returncode=1,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_pytest.invoke({})
    assert "Exit code: 1" in result
    assert "FAILED tests/test_math.py::test_add" in result
    assert "1 failed, 2 passed" in result


def test_run_pytest_captures_stderr(monkeypatch):
    def fake_run(*args, **kwargs):
        return _FakeCompletedProcess(
            stdout="", stderr="ImportError: no module named foo\n", returncode=2
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_pytest.invoke({})
    assert "Exit code: 2" in result
    assert "ImportError: no module named foo" in result


def test_run_pytest_uses_real_subprocess_timeout_constant(monkeypatch):
    captured_kwargs = {}

    def fake_run(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeCompletedProcess(stdout="1 passed\n", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_pytest.invoke({})
    assert captured_kwargs["timeout"] == tools._SUBPROCESS_TIMEOUT_SECONDS


def test_run_pytest_reports_timeout_without_claiming_success(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="pytest", timeout=tools._SUBPROCESS_TIMEOUT_SECONDS
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_pytest.invoke({})
    assert "too long" in result.lower()
    assert "passed" not in result.lower()


def test_run_pytest_handles_missing_executable_safely(monkeypatch):
    def fake_run(*args, **kwargs):
        raise OSError("pytest executable not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_pytest.invoke({})
    assert "error" in result.lower()
    assert "pytest executable not found" in result


def test_run_pytest_reports_missing_tests_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "PROJECT_ROOT", tmp_path)
    result = run_pytest.invoke({})
    assert "no tests/ folder" in result.lower()


# ---------------------------------------------------------------------------
# check_python_syntax - real parsing, no subprocess involved
# ---------------------------------------------------------------------------


def test_check_python_syntax_reports_no_errors_for_valid_file():
    result = check_python_syntax.invoke({"file_path": "conftest.py"})
    assert "no syntax errors found" in result.lower()


def test_check_python_syntax_detects_real_syntax_error():
    scratch = PROJECT_ROOT / "_scratch_bad_syntax_test.py"
    scratch.write_text("def broken(:\n    pass\n")
    try:
        result = check_python_syntax.invoke({"file_path": scratch.name})
        assert "syntax error" in result.lower()
        assert scratch.name in result
    finally:
        scratch.unlink()


def test_check_python_syntax_reports_real_line_number():
    scratch = PROJECT_ROOT / "_scratch_bad_syntax_line_test.py"
    scratch.write_text("x = 1\ny = 2\ndef broken(:\n    pass\n")
    try:
        result = check_python_syntax.invoke({"file_path": scratch.name})
        assert f"{scratch.name}:3" in result
    finally:
        scratch.unlink()


def test_check_python_syntax_checks_whole_project_with_no_errors():
    result = check_python_syntax.invoke({"file_path": "."})
    assert "no syntax errors found" in result.lower()


def test_check_python_syntax_rejects_path_traversal():
    result = check_python_syntax.invoke({"file_path": "../outside.py"})
    assert "outside the project directory" in result


def test_check_python_syntax_missing_file():
    result = check_python_syntax.invoke({"file_path": "does_not_exist.py"})
    assert "not found" in result.lower()


def test_check_python_syntax_rejects_dot_env():
    result = check_python_syntax.invoke({"file_path": ".env"})
    assert "error" in result.lower()
    assert "GOOGLE_API_KEY" not in result


def test_check_python_syntax_rejects_non_python_file():
    result = check_python_syntax.invoke({"file_path": "README.md"})
    assert "not a python file" in result.lower()


def test_check_python_syntax_default_argument_checks_whole_project():
    result = check_python_syntax.invoke({})
    assert "no syntax errors found" in result.lower()
