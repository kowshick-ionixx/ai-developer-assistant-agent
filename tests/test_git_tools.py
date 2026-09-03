"""
Tests for the Phase 4 git_status / git_log / git_diff / git_branch tools in
tools.py.

GitPython's Repo is mocked at the tools._get_repo() boundary (the same
pattern test_agent.py uses to fake the LangChain agent) so these tests never
need a real `git` executable or a real repository on disk.
"""

import datetime

from git.exc import GitCommandError

import tools
from tools import git_branch, git_diff, git_log, git_status


class _FakeAuthor:
    def __init__(self, name):
        self.name = name


class _FakeCommit:
    def __init__(self, hexsha, message, author_name, when):
        self.hexsha = hexsha
        self.message = message
        self.author = _FakeAuthor(author_name)
        self.committed_datetime = when


class _FakeBranch:
    def __init__(self, name):
        self.name = name


class _FakeHead:
    def __init__(self, is_detached):
        self.is_detached = is_detached


class _FakeActiveBranch:
    def __init__(self, name):
        self.name = name


class _FakeGitCLI:
    """Stands in for GitPython's `repo.git` command proxy."""

    def __init__(
        self, status="", diff="", diff_cached="", diff_stat="", diff_error=None
    ):
        self._status = status
        self._diff = diff
        self._diff_cached = diff_cached
        self._diff_stat = diff_stat
        self._diff_error = diff_error
        self.diff_calls = []

    def version(self):
        return "git version 2.44.0"

    def status(self, *args):
        return self._status

    def diff(self, *args):
        self.diff_calls.append(args)
        if self._diff_error is not None:
            raise self._diff_error
        if "--stat" in args:
            return self._diff_stat
        if "--cached" in args:
            return self._diff_cached
        return self._diff


class _FakeRepo:
    def __init__(
        self,
        git_cli=None,
        commits=None,
        branches=None,
        active_branch_name="main",
        detached=False,
    ):
        self.git = git_cli or _FakeGitCLI()
        self._commits = commits or []
        self.branches = branches if branches is not None else [_FakeBranch("main")]
        self.head = _FakeHead(detached)
        self._active_branch_name = active_branch_name
        self.iter_commits_calls = []

    def iter_commits(self, max_count=10):
        self.iter_commits_calls.append(max_count)
        return self._commits[:max_count]

    @property
    def active_branch(self):
        return _FakeActiveBranch(self._active_branch_name)


def _use_repo(monkeypatch, repo):
    monkeypatch.setattr(tools, "_get_repo", lambda: repo)


# ---------------------------------------------------------------------------
# Not a Git repository
# ---------------------------------------------------------------------------


def test_git_status_reports_missing_repository(monkeypatch):
    monkeypatch.setattr(tools, "_get_repo", lambda: None)
    result = git_status.invoke({})
    assert "not" in result.lower()
    assert "git" in result.lower()


def test_git_log_reports_missing_repository(monkeypatch):
    monkeypatch.setattr(tools, "_get_repo", lambda: None)
    assert "git" in git_log.invoke({}).lower()


def test_git_diff_reports_missing_repository(monkeypatch):
    monkeypatch.setattr(tools, "_get_repo", lambda: None)
    assert "git" in git_diff.invoke({}).lower()


def test_git_branch_reports_missing_repository(monkeypatch):
    monkeypatch.setattr(tools, "_get_repo", lambda: None)
    assert "git" in git_branch.invoke({}).lower()


# ---------------------------------------------------------------------------
# git_status
# ---------------------------------------------------------------------------


def test_git_status_clean_working_tree(monkeypatch):
    _use_repo(monkeypatch, _FakeRepo(_FakeGitCLI(status="")))
    result = git_status.invoke({})
    assert "clean" in result.lower()


def test_git_status_categorizes_changes(monkeypatch):
    porcelain = " M agent.py\nA  new_file.py\n?? untracked.py\n D deleted.py\n"
    _use_repo(monkeypatch, _FakeRepo(_FakeGitCLI(status=porcelain)))
    result = git_status.invoke({})
    assert "agent.py" in result and "Modified:" in result
    assert "new_file.py" in result and "Staged:" in result
    assert "untracked.py" in result and "Untracked:" in result
    assert "deleted.py" in result and "Deleted:" in result


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------


def test_git_log_formats_recent_commits(monkeypatch):
    when = datetime.datetime(2026, 1, 15, tzinfo=datetime.timezone.utc)
    commits = [
        _FakeCommit("abc1234567", "Add project search\n\nlong body", "Ada", when),
        _FakeCommit("def8901234", "Improve CLI logging", "Bob", when),
    ]
    _use_repo(monkeypatch, _FakeRepo(commits=commits))
    result = git_log.invoke({})
    assert "abc1234" in result
    assert "Add project search" in result
    assert "long body" not in result
    assert "Ada" in result and "2026-01-15" in result


def test_git_log_no_commits(monkeypatch):
    _use_repo(monkeypatch, _FakeRepo(commits=[]))
    result = git_log.invoke({})
    assert "no commits" in result.lower()


def test_git_log_caps_max_count(monkeypatch):
    repo = _FakeRepo(commits=[])
    _use_repo(monkeypatch, repo)
    git_log.invoke({"max_count": 9999})
    assert repo.iter_commits_calls == [tools._MAX_GIT_LOG_COMMITS]


def test_git_log_clamps_out_of_range_max_count(monkeypatch):
    repo = _FakeRepo(commits=[])
    _use_repo(monkeypatch, repo)
    git_log.invoke({"max_count": 0})
    assert repo.iter_commits_calls == [1]


# ---------------------------------------------------------------------------
# git_diff
# ---------------------------------------------------------------------------


def test_git_diff_no_changes(monkeypatch):
    _use_repo(monkeypatch, _FakeRepo(_FakeGitCLI(diff="", diff_cached="")))
    result = git_diff.invoke({})
    assert "no changes" in result.lower()


def test_git_diff_returns_diff_text(monkeypatch):
    diff_text = "diff --git a/agent.py b/agent.py\n+added line\n"
    _use_repo(monkeypatch, _FakeRepo(_FakeGitCLI(diff=diff_text)))
    result = git_diff.invoke({})
    assert result == diff_text


def test_git_diff_falls_back_to_staged(monkeypatch):
    staged_diff = "diff --git a/tools.py b/tools.py\n+staged change\n"
    _use_repo(monkeypatch, _FakeRepo(_FakeGitCLI(diff="", diff_cached=staged_diff)))
    result = git_diff.invoke({})
    assert result == staged_diff


def test_git_diff_summarizes_when_too_large(monkeypatch):
    huge_diff = "diff --git a/big.py b/big.py\n" + ("+line\n" * 5000)
    stat = "big.py | 5000 +++++++++\n"
    _use_repo(
        monkeypatch,
        _FakeRepo(_FakeGitCLI(diff=huge_diff, diff_stat=stat)),
    )
    result = git_diff.invoke({})
    assert "too large" in result.lower()
    assert "big.py" in result


def test_git_diff_redacts_blocked_files(monkeypatch):
    diff_text = (
        "diff --git a/.env b/.env\n"
        "+GOOGLE_API_KEY=AIzaSyD-fake1234567890abcdefghijklmno\n"
        "diff --git a/agent.py b/agent.py\n"
        "+normal change\n"
    )
    _use_repo(monkeypatch, _FakeRepo(_FakeGitCLI(diff=diff_text)))
    result = git_diff.invoke({})
    assert "AIza" not in result
    assert "GOOGLE_API_KEY" not in result
    assert "REDACTED" in result
    assert "normal change" in result


def test_git_diff_handles_git_command_error(monkeypatch):
    cli = _FakeGitCLI(diff_error=GitCommandError("diff", 128, stderr="bad path"))
    _use_repo(monkeypatch, _FakeRepo(cli))
    result = git_diff.invoke({"file_path": "does/not/exist.py"})
    assert "error" in result.lower()


# ---------------------------------------------------------------------------
# git_branch
# ---------------------------------------------------------------------------


def test_git_branch_reports_current_and_local_branches(monkeypatch):
    branches = [_FakeBranch("main"), _FakeBranch("feature/phase-4")]
    _use_repo(monkeypatch, _FakeRepo(branches=branches, active_branch_name="main"))
    result = git_branch.invoke({})
    assert "Current branch: main" in result
    assert "feature/phase-4" in result


def test_git_branch_handles_detached_head(monkeypatch):
    _use_repo(monkeypatch, _FakeRepo(detached=True))
    result = git_branch.invoke({})
    assert "detached" in result.lower()
