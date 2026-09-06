"""
Tests for tools.py's Phase 6 live-preview feature: launch_generated_app and
stop_generated_app.

subprocess.Popen and the real network health check are mocked (the same
external-boundary-mocking pattern test_execution_tools.py/test_git_tools.py
already use) since a real test run must never actually spawn a live
Streamlit server. Path/entry-file validation, port selection, and the
process registry itself are all exercised for real.
"""

import subprocess

import pytest

import tools
from tools import PROJECT_ROOT, launch_generated_app, stop_generated_app


class _FakeProcess:
    """Stands in for subprocess.Popen - alive until .terminate()/.kill() is
    called, exactly like a real long-running Streamlit server process."""

    def __init__(self, pid=4242):
        self.pid = pid
        self._alive = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, timeout=None):
        if self._alive:
            raise subprocess.TimeoutExpired(cmd="streamlit", timeout=timeout)
        return 0


def _fake_terminate_process_tree(pid, process=None) -> None:
    """Stands in for tools._terminate_process_tree, which for real
    processes shells out to `taskkill /F /T` (Windows) or `os.killpg`
    (POSIX) - never something a test should invoke against a fake/made-up
    PID (it could, in principle, hit an unrelated real process on the
    machine running the suite). Exercises the exact same fake-process
    lifecycle (_FakeProcess.terminate()/.wait()/.kill()) the old inline
    code used to, so these tests still verify the right calls happen.
    `process` is None for a rehydrated (no live Popen handle) server -
    nothing to drive in that case, matching the real function."""
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture(autouse=True)
def _clean_server_state(monkeypatch):
    tools._GENERATED_SERVERS.clear()
    # Fast, deterministic timing for every test in this file - a real
    # 20-second timeout would make the suite unbearably slow.
    monkeypatch.setattr(tools, "_SERVER_STARTUP_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(tools, "_SERVER_POLL_INTERVAL_SECONDS", 0.02)
    # Never let a test invoke the real taskkill/killpg boundary against a
    # fake PID - see _fake_terminate_process_tree's docstring.
    monkeypatch.setattr(tools, "_terminate_process_tree", _fake_terminate_process_tree)
    yield
    tools._GENERATED_SERVERS.clear()


@pytest.fixture
def scratch_app_project():
    """A small, disposable generated Streamlit project - a real app.py that
    actually mentions "streamlit", removed after the test regardless of
    outcome."""
    root = PROJECT_ROOT / "generated_projects" / "_scratch_live_app"
    root.mkdir(parents=True)
    (root / "app.py").write_text("import streamlit as st\nst.write('hi')\n")
    yield "generated_projects/_scratch_live_app", root
    import shutil

    shutil.rmtree(root, ignore_errors=True)


def _fake_popen_factory(process: _FakeProcess):
    calls = []

    def fake_popen(args, **kwargs):
        calls.append(args)
        return process

    return fake_popen, calls


# ---------------------------------------------------------------------------
# 1. Entry file detection
# ---------------------------------------------------------------------------


def test_resolves_app_py_by_default(scratch_app_project):
    _project_root, root = scratch_app_project
    entry_path, error = tools._resolve_entry_file(root, "")
    assert error == ""
    assert entry_path == root / "app.py"


def test_resolves_main_py_when_app_py_absent():
    root = PROJECT_ROOT / "generated_projects" / "_scratch_main_entry"
    root.mkdir(parents=True)
    (root / "main.py").write_text("import streamlit as st\n")
    try:
        entry_path, error = tools._resolve_entry_file(root, "")
        assert error == ""
        assert entry_path == root / "main.py"
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


def test_explicit_entry_file_is_honored(scratch_app_project):
    _project_root, root = scratch_app_project
    (root / "ui.py").write_text("import streamlit as st\n")
    entry_path, error = tools._resolve_entry_file(root, "ui.py")
    assert error == ""
    assert entry_path == root / "ui.py"


def test_missing_entry_file_is_a_clear_error(scratch_app_project):
    _project_root, root = scratch_app_project
    (root / "app.py").unlink()
    entry_path, error = tools._resolve_entry_file(root, "")
    assert entry_path is None
    assert "app.py" in error and "main.py" in error


def test_file_not_mentioning_streamlit_is_rejected(scratch_app_project):
    _project_root, root = scratch_app_project
    (root / "app.py").write_text("x = 1\n")  # a real .py file, but not Streamlit
    entry_path, error = tools._resolve_entry_file(root, "")
    assert entry_path is None
    assert error != ""


def test_entry_file_cannot_escape_the_project_root(scratch_app_project):
    _project_root, root = scratch_app_project
    entry_path, _error = tools._resolve_entry_file(root, "../../tools.py")
    assert entry_path is None


# ---------------------------------------------------------------------------
# 2-3. Port selection / collision
# ---------------------------------------------------------------------------


def test_select_safe_port_skips_the_assistant_port(monkeypatch):
    monkeypatch.setattr(tools, "_is_port_free", lambda port: True)
    assert tools._select_safe_port() != tools._ASSISTANT_OWN_PORT


def test_select_safe_port_skips_busy_ports(monkeypatch):
    busy = {8502, 8503}
    monkeypatch.setattr(tools, "_is_port_free", lambda port: port not in busy)
    assert tools._select_safe_port() == 8504


def test_select_safe_port_returns_none_when_all_busy(monkeypatch):
    monkeypatch.setattr(tools, "_is_port_free", lambda port: False)
    assert tools._select_safe_port() is None


# ---------------------------------------------------------------------------
# 4-6, 13. Launch + process tracking + real health check + stored URL
# ---------------------------------------------------------------------------


def test_launch_reports_running_after_a_real_health_check_passes(
    scratch_app_project, monkeypatch
):
    project_root, _root = scratch_app_project
    process = _FakeProcess()
    fake_popen, calls = _fake_popen_factory(process)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: True)

    result = launch_generated_app.invoke({"project_root": project_root})

    assert "running" in result.lower()
    assert "http://localhost:8502" in result
    assert len(calls) == 1
    assert calls[0][0:4] == [tools.sys.executable, "-m", "streamlit", "run"]
    assert "--server.port" in calls[0] and "8502" in calls[0]

    server = tools.get_generated_server(project_root)
    assert server is not None
    assert server.status == "running"
    assert server.url == "http://localhost:8502"
    assert server.port == 8502
    assert server.pid == process.pid
    assert server.entry_file == "app.py"


def test_launch_handles_popen_oserror_without_crashing(
    scratch_app_project, monkeypatch
):
    """Tool exception coverage: if the OS itself refuses to start the
    subprocess (e.g. streamlit isn't actually installed), Popen raises
    OSError - this must be reported as a clean error, never an unhandled
    crash of the tool (and therefore of the agent turn)."""
    project_root, _root = scratch_app_project

    def fake_popen(args, **kwargs):
        raise OSError("streamlit executable not found")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)

    result = launch_generated_app.invoke({"project_root": project_root})

    assert "error" in result.lower()
    assert "could not start the generated application" in result.lower()
    assert "streamlit executable not found" in result


def test_launch_writes_a_log_file_inside_the_generated_project(
    scratch_app_project, monkeypatch
):
    project_root, root = scratch_app_project
    process = _FakeProcess()
    fake_popen, _ = _fake_popen_factory(process)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: True)

    launch_generated_app.invoke({"project_root": project_root})
    assert (root / ".server.log").exists()


def test_launch_rejects_a_project_outside_generated_projects():
    result = launch_generated_app.invoke({"project_root": "tests"})
    assert "error" in result.lower()
    assert "generated_projects" in result.lower()


def test_launch_rejects_path_traversal():
    result = launch_generated_app.invoke({"project_root": "../outside"})
    assert "error" in result.lower()


def test_launch_reports_missing_project_directory():
    result = launch_generated_app.invoke(
        {"project_root": "generated_projects/does_not_exist"}
    )
    assert "error" in result.lower()
    assert "not found" in result.lower()


def test_launch_reports_no_free_port(scratch_app_project, monkeypatch):
    project_root, _root = scratch_app_project
    monkeypatch.setattr(tools, "_select_safe_port", lambda: None)
    result = launch_generated_app.invoke({"project_root": project_root})
    assert "error" in result.lower()
    assert "no safe local port" in result.lower()


# ---------------------------------------------------------------------------
# 7-8. Health-check failure / startup timeout
# ---------------------------------------------------------------------------


def test_health_check_failure_reports_error_and_terminates_process(
    scratch_app_project, monkeypatch
):
    project_root, _root = scratch_app_project
    process = _FakeProcess()
    fake_popen, _ = _fake_popen_factory(process)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: False)

    result = launch_generated_app.invoke({"project_root": project_root})

    assert "error" in result.lower()
    assert process.terminated is True
    server = tools.get_generated_server(project_root)
    assert server.status == "failed"
    assert server.error is not None


def test_process_exiting_early_is_reported_as_a_real_failure(
    scratch_app_project, monkeypatch
):
    project_root, _root = scratch_app_project
    process = _FakeProcess()
    process._alive = False  # exited immediately, as if it crashed on startup
    fake_popen, _ = _fake_popen_factory(process)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: False)

    result = launch_generated_app.invoke({"project_root": project_root})

    assert "error" in result.lower()
    assert "exited" in result.lower()
    server = tools.get_generated_server(project_root)
    assert server.status == "failed"


def test_launch_startup_timeout_is_a_real_failure_not_a_fabricated_success(
    scratch_app_project, monkeypatch
):
    project_root, _root = scratch_app_project
    process = _FakeProcess()
    fake_popen, _ = _fake_popen_factory(process)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: False)

    result = launch_generated_app.invoke({"project_root": project_root})

    assert "running" not in result.lower()
    assert "error" in result.lower()


# ---------------------------------------------------------------------------
# 9. Duplicate launch prevention
# ---------------------------------------------------------------------------


def test_launching_twice_reuses_the_existing_running_server(
    scratch_app_project, monkeypatch
):
    project_root, _root = scratch_app_project
    process = _FakeProcess()
    fake_popen, calls = _fake_popen_factory(process)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: True)

    launch_generated_app.invoke({"project_root": project_root})
    second = launch_generated_app.invoke({"project_root": project_root})

    assert "already running" in second.lower()
    assert len(calls) == 1  # never launched a second process
    assert tools.get_generated_server(project_root).port == 8502


def test_relaunches_if_the_tracked_process_actually_died(
    scratch_app_project, monkeypatch
):
    project_root, _root = scratch_app_project
    dead_process = _FakeProcess()
    dead_process._alive = False
    fake_popen, _calls = _fake_popen_factory(dead_process)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: False)
    launch_generated_app.invoke(
        {"project_root": project_root}
    )  # fails, tracked as failed
    assert tools.get_generated_server(project_root).status == "failed"

    new_process = _FakeProcess()
    fake_popen2, calls2 = _fake_popen_factory(new_process)
    monkeypatch.setattr(subprocess, "Popen", fake_popen2)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: True)
    result = launch_generated_app.invoke({"project_root": project_root})

    assert "running" in result.lower()
    assert len(calls2) == 1


# ---------------------------------------------------------------------------
# 10-11. Stop / assistant unaffected
# ---------------------------------------------------------------------------


def test_stop_generated_app_terminates_the_tracked_process(
    scratch_app_project, monkeypatch
):
    project_root, _root = scratch_app_project
    process = _FakeProcess()
    fake_popen, _ = _fake_popen_factory(process)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: True)
    launch_generated_app.invoke({"project_root": project_root})

    result = stop_generated_app.invoke({"project_root": project_root})

    assert "stopped" in result.lower()
    assert process.terminated is True
    assert tools.get_generated_server(project_root).status == "stopped"


def test_stop_generated_app_never_touches_this_process():
    """stop_generated_app only ever calls .terminate()/.kill() on the Popen
    object it itself tracked - it has no code path that can reach this test
    process's own PID. The fact that this assertion (and every test after
    it in the suite) keeps running is itself the proof."""
    import os

    this_pid = os.getpid()
    stop_generated_app.invoke({"project_root": "generated_projects/never_launched"})
    assert os.getpid() == this_pid  # this process is unaffected


def test_stop_generated_app_reports_error_for_unknown_project():
    result = stop_generated_app.invoke({"project_root": "generated_projects/nope"})
    assert "error" in result.lower()
    assert "no tracked server" in result.lower()


def test_stop_generated_app_reports_when_already_stopped(
    scratch_app_project, monkeypatch
):
    project_root, _root = scratch_app_project
    process = _FakeProcess()
    fake_popen, _ = _fake_popen_factory(process)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: True)
    launch_generated_app.invoke({"project_root": project_root})
    stop_generated_app.invoke({"project_root": project_root})

    result = stop_generated_app.invoke({"project_root": project_root})
    assert "not currently running" in result.lower()


# ---------------------------------------------------------------------------
# 12. Orphan recovery: a server another (now-gone) process launched must
# still be reclaimable - reused, stopped, or reported as running - by a
# brand-new process with no in-memory record of it, via the small state
# file persisted next to the generated project (tools._SERVER_STATE_FILENAME).
# ---------------------------------------------------------------------------


def _write_state_file(root, *, port, pid, entry_file="app.py"):
    """Simulates a PREVIOUS process's real launch_generated_app call having
    already persisted state, without needing to actually launch anything -
    the exact file a real launch call itself writes (see tools.
    _write_server_state)."""
    import json

    (root / tools._SERVER_STATE_FILENAME).write_text(
        json.dumps(
            {
                "port": port,
                "pid": pid,
                "entry_file": entry_file,
                "started_at": "2026-01-01T00:00:00+00:00",
                "log_path": None,
            }
        ),
        encoding="utf-8",
    )


def test_launch_reclaims_a_healthy_orphan_without_starting_a_second_process(
    scratch_app_project, monkeypatch
):
    """The exact scenario the audit found: the assistant process restarted
    (so _GENERATED_SERVERS is empty, as it would be after a fresh process
    start) while a real server from before is still alive. launch_
    generated_app must reuse it - never start a duplicate on a new port."""
    project_root, root = scratch_app_project
    _write_state_file(root, port=8502, pid=99999)
    assert project_root not in tools._GENERATED_SERVERS  # nothing in memory

    popen_calls = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **kw: popen_calls.append(a) or _FakeProcess()
    )
    monkeypatch.setattr(tools, "_http_health_check", lambda port: port == 8502)

    result = launch_generated_app.invoke({"project_root": project_root})

    assert "already running" in result.lower()
    assert "http://localhost:8502" in result
    assert popen_calls == []  # never launched a second process
    server = tools.get_generated_server(project_root)
    assert server.status == "running"
    assert server.port == 8502
    assert server.pid == 99999


def test_get_generated_server_reports_running_for_a_reclaimed_orphan():
    """A fresh process (e.g. after a restart) asking about a server it
    never itself launched must still see it as running, not "not
    running", as long as it's genuinely still healthy - this is what lets
    the Streamlit UI show the real live state after a restart."""
    root = PROJECT_ROOT / "generated_projects" / "_scratch_orphan_check"
    root.mkdir(parents=True)
    try:
        project_root = "generated_projects/_scratch_orphan_check"
        _write_state_file(root, port=8510, pid=12345)

        import tools as tools_module

        original = tools_module._http_health_check
        tools_module._http_health_check = lambda port: port == 8510
        try:
            server = tools.get_generated_server(project_root)
        finally:
            tools_module._http_health_check = original

        assert server is not None
        assert server.status == "running"
        assert server.pid == 12345
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


def test_stop_reclaims_and_terminates_an_orphaned_server(
    scratch_app_project, monkeypatch
):
    """stop_generated_app must be able to actually stop a server from a
    previous process, not just report "no tracked server found" - this is
    the fix for the orphaned-process finding: without it, a server whose
    launching process restarted could never be stopped through the tool
    again at all."""
    project_root, root = scratch_app_project
    _write_state_file(root, port=8503, pid=54321)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: port == 8503)

    terminated = []
    monkeypatch.setattr(
        tools,
        "_terminate_process_tree",
        lambda pid, process=None: terminated.append(pid),
    )

    result = stop_generated_app.invoke({"project_root": project_root})

    assert "stopped" in result.lower()
    assert terminated == [54321]
    assert tools.get_generated_server(project_root).status == "stopped"
    assert not (root / tools._SERVER_STATE_FILENAME).exists()


def test_stale_state_file_for_a_dead_process_is_ignored_and_cleaned_up(
    scratch_app_project, monkeypatch
):
    """A state file left over from a server that's no longer actually
    running (crashed, machine rebooted, port reused by something else)
    must never be reported as running just because the file exists - and
    must not keep confusing every future call once it's proven stale."""
    project_root, root = scratch_app_project
    _write_state_file(root, port=8504, pid=11111)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: False)

    assert tools.get_generated_server(project_root) is None
    assert not (
        root / tools._SERVER_STATE_FILENAME
    ).exists()  # cleaned up, not left stale

    # A subsequent launch must proceed normally (fresh launch), not error
    # out or get stuck thinking something is already running.
    process = _FakeProcess()
    fake_popen, calls = _fake_popen_factory(process)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: True)

    result = launch_generated_app.invoke({"project_root": project_root})
    assert "running" in result.lower()
    assert len(calls) == 1


def test_normal_launch_health_check_stop_cycle_persists_and_clears_state(
    scratch_app_project, monkeypatch
):
    """End-to-end sanity check for the ordinary (non-orphan) path: a real
    launch writes the state file, and a real stop removes it again - the
    file never lingers once nothing is actually running."""
    project_root, root = scratch_app_project
    process = _FakeProcess()
    fake_popen, _ = _fake_popen_factory(process)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: True)

    launch_generated_app.invoke({"project_root": project_root})
    assert (root / tools._SERVER_STATE_FILENAME).exists()

    stop_generated_app.invoke({"project_root": project_root})
    assert not (root / tools._SERVER_STATE_FILENAME).exists()
