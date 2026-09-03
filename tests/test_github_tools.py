"""
Tests for the Phase 4 github_get_repository / github_get_issues /
github_get_pull_requests tools in tools.py.

PyGithub is mocked at the tools._get_github_client() boundary so these tests
never make a real network call or require a real GITHUB_TOKEN.
"""

import datetime

from github import GithubException

import tools
from tools import github_get_issues, github_get_pull_requests, github_get_repository


class _FakeUser:
    def __init__(self, login):
        self.login = login


class _FakeIssue:
    def __init__(self, number, title, login, state, created_at, is_pr=False):
        self.number = number
        self.title = title
        self.user = _FakeUser(login)
        self.state = state
        self.created_at = created_at
        self.pull_request = object() if is_pr else None


class _FakePullRequest:
    def __init__(self, number, title, login, state, created_at):
        self.number = number
        self.title = title
        self.user = _FakeUser(login)
        self.state = state
        self.created_at = created_at


class _FakeGithubRepo:
    def __init__(
        self,
        full_name="octocat/hello-world",
        description="A test repository",
        default_branch="main",
        language="Python",
        stars=42,
        open_issues=3,
        url="https://github.com/octocat/hello-world",
        issues=None,
        pulls=None,
    ):
        self.full_name = full_name
        self.description = description
        self.default_branch = default_branch
        self.language = language
        self.stargazers_count = stars
        self.open_issues_count = open_issues
        self.html_url = url
        self._issues = issues or []
        self._pulls = pulls or []

    def get_issues(self, state="open"):
        return self._issues

    def get_pulls(self, state="open"):
        return self._pulls


class _FakeGithubClient:
    def __init__(self, repo=None, error=None):
        self._repo = repo
        self._error = error

    def get_repo(self, name):
        if self._error is not None:
            raise self._error
        return self._repo


def _use_client(monkeypatch, client):
    monkeypatch.setattr(tools, "_get_github_client", lambda: client)


WHEN = datetime.datetime(2026, 2, 1, tzinfo=datetime.timezone.utc)


# ---------------------------------------------------------------------------
# No repository specified
# ---------------------------------------------------------------------------


def test_github_get_repository_requires_repo_name(monkeypatch):
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    result = github_get_repository.invoke({"repo_full_name": ""})
    assert "no repository specified" in result.lower()


def test_github_get_repository_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "octocat/hello-world")
    _use_client(monkeypatch, _FakeGithubClient(repo=_FakeGithubRepo()))
    result = github_get_repository.invoke({"repo_full_name": ""})
    assert "octocat/hello-world" in result


# ---------------------------------------------------------------------------
# github_get_repository
# ---------------------------------------------------------------------------


def test_github_get_repository_success(monkeypatch):
    _use_client(monkeypatch, _FakeGithubClient(repo=_FakeGithubRepo()))
    result = github_get_repository.invoke({"repo_full_name": "octocat/hello-world"})
    assert "octocat/hello-world" in result
    assert "A test repository" in result
    assert "main" in result
    assert "42" in result


def test_github_get_repository_not_found(monkeypatch):
    _use_client(
        monkeypatch,
        _FakeGithubClient(error=GithubException(404, {"message": "Not Found"})),
    )
    result = github_get_repository.invoke({"repo_full_name": "octocat/missing"})
    assert "not found" in result.lower()


def test_github_get_repository_auth_failure(monkeypatch):
    _use_client(
        monkeypatch,
        _FakeGithubClient(error=GithubException(401, {"message": "Bad credentials"})),
    )
    result = github_get_repository.invoke({"repo_full_name": "octocat/private"})
    assert "authentication failed" in result.lower() or "rate limit" in result.lower()


def test_github_get_repository_network_failure(monkeypatch):
    class _BoomClient:
        def get_repo(self, name):
            raise ConnectionError("simulated network failure")

    _use_client(monkeypatch, _BoomClient())
    result = github_get_repository.invoke({"repo_full_name": "octocat/hello-world"})
    assert "error" in result.lower()


# ---------------------------------------------------------------------------
# github_get_issues
# ---------------------------------------------------------------------------


def test_github_get_issues_requires_repo_name(monkeypatch):
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    result = github_get_issues.invoke({"repo_full_name": ""})
    assert "no repository specified" in result.lower()


def test_github_get_issues_rejects_invalid_state():
    result = github_get_issues.invoke(
        {"repo_full_name": "octocat/hello-world", "state": "bogus"}
    )
    assert "error" in result.lower()


def test_github_get_issues_filters_out_pull_requests(monkeypatch):
    issues = [
        _FakeIssue(21, "Real issue", "alice", "open", WHEN),
        _FakeIssue(22, "This is actually a PR", "bob", "open", WHEN, is_pr=True),
        _FakeIssue(23, "Another real issue", "carol", "open", WHEN),
    ]
    _use_client(
        monkeypatch,
        _FakeGithubClient(repo=_FakeGithubRepo(issues=issues)),
    )
    result = github_get_issues.invoke({"repo_full_name": "octocat/hello-world"})
    assert "#21" in result and "Real issue" in result
    assert "#23" in result and "Another real issue" in result
    assert "#22" not in result


def test_github_get_issues_no_issues(monkeypatch):
    _use_client(monkeypatch, _FakeGithubClient(repo=_FakeGithubRepo(issues=[])))
    result = github_get_issues.invoke({"repo_full_name": "octocat/hello-world"})
    assert "no open issues" in result.lower()


def test_github_get_issues_not_found(monkeypatch):
    _use_client(
        monkeypatch,
        _FakeGithubClient(error=GithubException(404, {"message": "Not Found"})),
    )
    result = github_get_issues.invoke({"repo_full_name": "octocat/missing"})
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# github_get_pull_requests
# ---------------------------------------------------------------------------


def test_github_get_pull_requests_success(monkeypatch):
    pulls = [_FakePullRequest(7, "Add Phase 4 tools", "dave", "open", WHEN)]
    _use_client(
        monkeypatch,
        _FakeGithubClient(repo=_FakeGithubRepo(pulls=pulls)),
    )
    result = github_get_pull_requests.invoke({"repo_full_name": "octocat/hello-world"})
    assert "#7" in result
    assert "Add Phase 4 tools" in result
    assert "dave" in result


def test_github_get_pull_requests_no_results(monkeypatch):
    _use_client(monkeypatch, _FakeGithubClient(repo=_FakeGithubRepo(pulls=[])))
    result = github_get_pull_requests.invoke({"repo_full_name": "octocat/hello-world"})
    assert "no open pull requests" in result.lower()


def test_github_get_pull_requests_requires_repo_name(monkeypatch):
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    result = github_get_pull_requests.invoke({"repo_full_name": ""})
    assert "no repository specified" in result.lower()


# ---------------------------------------------------------------------------
# Token / secret protection
# ---------------------------------------------------------------------------


def test_get_github_client_uses_token_without_leaking_it(monkeypatch):
    fake_token = "ghp_fakeTokenForTestsOnly1234567890"
    monkeypatch.setenv("GITHUB_TOKEN", fake_token)

    captured = {}

    class _FakeAuthToken:
        def __init__(self, token):
            captured["token"] = token

    class _FakeGithub:
        def __init__(self, auth=None):
            captured["auth"] = auth

    monkeypatch.setattr(tools, "Auth", type("Auth", (), {"Token": _FakeAuthToken}))
    monkeypatch.setattr(tools, "Github", _FakeGithub)

    client = tools._get_github_client()

    assert captured["token"] == fake_token
    assert isinstance(client, _FakeGithub)
