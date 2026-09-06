"""
Tests for the Phase 4 web_search / documentation_search tools in tools.py.

Tavily is mocked (via tools.TavilyClient) so these tests never make a real
network call or require a real TAVILY_API_KEY.
"""

import pytest

import tools
from tools import documentation_search, web_search


class _FakeTavilyClient:
    """Records the search() call and returns canned results."""

    last_call = None

    def __init__(self, api_key):
        self.api_key = api_key

    def search(self, query, max_results, search_depth, include_domains):
        _FakeTavilyClient.last_call = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_domains": include_domains,
        }
        return {
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.com/{i}",
                    "content": f"Content for result {i}. " * 50,
                }
                for i in range(1, 8)
            ]
        }


class _EmptyTavilyClient:
    def __init__(self, api_key):
        pass

    def search(self, **kwargs):
        return {"results": []}


class _FailingTavilyClient:
    def __init__(self, api_key):
        pass

    def search(self, **kwargs):
        raise TimeoutError("simulated network timeout")


@pytest.fixture(autouse=True)
def _tavily_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-key-for-tests")


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


def test_web_search_requires_query():
    result = web_search.invoke({"query": "  "})
    assert "error" in result.lower()


def test_web_search_missing_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = web_search.invoke({"query": "current LangChain agent API"})
    assert "TAVILY_API_KEY" in result
    assert "not configured" in result.lower()


def test_web_search_returns_results_labeled_as_untrusted(monkeypatch):
    monkeypatch.setattr(tools, "TavilyClient", _FakeTavilyClient)
    result = web_search.invoke({"query": "current LangChain agent API"})
    assert "Result 1" in result
    assert "https://example.com/1" in result
    assert "untrusted" in result.lower()


def test_web_search_limits_result_count(monkeypatch):
    monkeypatch.setattr(tools, "TavilyClient", _FakeTavilyClient)
    result = web_search.invoke({"query": "python"})
    assert result.count("Source:") == tools._MAX_WEB_RESULTS


def test_web_search_truncates_long_snippets(monkeypatch):
    monkeypatch.setattr(tools, "TavilyClient", _FakeTavilyClient)
    result = web_search.invoke({"query": "python"})
    assert "..." in result


def test_web_search_empty_results(monkeypatch):
    monkeypatch.setattr(tools, "TavilyClient", _EmptyTavilyClient)
    result = web_search.invoke({"query": "zzz_nonexistent_zzz"})
    assert "no results found" in result.lower()


def test_web_search_handles_api_failure(monkeypatch):
    monkeypatch.setattr(tools, "TavilyClient", _FailingTavilyClient)
    result = web_search.invoke({"query": "python"})
    assert "error" in result.lower()
    assert "search failed" in result.lower()


def test_web_search_handles_malformed_non_dict_response(monkeypatch):
    """Malformed tool output from an external dependency: Tavily's SDK is
    expected to return a dict, but _tavily_search only trusts that via
    isinstance(response, dict) - a response shaped differently (e.g. a bare
    list, or None) must degrade to "no results" instead of raising."""

    class _MalformedTavilyClient:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            return ["not", "a", "dict"]

    monkeypatch.setattr(tools, "TavilyClient", _MalformedTavilyClient)
    result = web_search.invoke({"query": "python"})
    assert "no results found" in result.lower()


def test_web_search_never_leaks_api_key(monkeypatch):
    monkeypatch.setattr(tools, "TavilyClient", _FakeTavilyClient)
    result = web_search.invoke({"query": "python"})
    assert "tvly-fake-key-for-tests" not in result


# ---------------------------------------------------------------------------
# documentation_search
# ---------------------------------------------------------------------------


def test_documentation_search_requires_query():
    result = documentation_search.invoke({"query": ""})
    assert "error" in result.lower()


def test_documentation_search_missing_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = documentation_search.invoke({"query": "pathlib"})
    assert "not configured" in result.lower()


def test_documentation_search_prefers_official_domain_for_known_topic(monkeypatch):
    monkeypatch.setattr(tools, "TavilyClient", _FakeTavilyClient)
    documentation_search.invoke(
        {"query": "Find the current Python pathlib documentation"}
    )
    assert _FakeTavilyClient.last_call["include_domains"] == ["docs.python.org"]


def test_documentation_search_falls_back_for_unknown_topic(monkeypatch):
    monkeypatch.setattr(tools, "TavilyClient", _FakeTavilyClient)
    result = documentation_search.invoke({"query": "some obscure unheard-of tool xyz"})
    assert _FakeTavilyClient.last_call["include_domains"] is None
    assert "no official source recognized" in result.lower()


def test_documentation_search_empty_results(monkeypatch):
    monkeypatch.setattr(tools, "TavilyClient", _EmptyTavilyClient)
    result = documentation_search.invoke({"query": "zzz_nonexistent_zzz"})
    assert "no results found" in result.lower()


def test_documentation_search_handles_network_failure(monkeypatch):
    monkeypatch.setattr(tools, "TavilyClient", _FailingTavilyClient)
    result = documentation_search.invoke({"query": "pathlib"})
    assert "error" in result.lower()
