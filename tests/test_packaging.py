"""
Tests for tools.py's Phase 6 final packaging step: create_project_zip and
its shared implementation, get_or_build_project_zip.

These exercise the real packaging logic directly against this project's
actual filesystem (the same way tests/test_dev_tools.py exercises
propose_file_change/apply_approved_change) - any scratch file created for a
test is cleaned up afterward, and the real "dist/" output folder this
feature writes into is cleared before and after every test in this file so
no test can see another test's leftover archive.
"""

import zipfile

import pytest

import tools
from tools import (
    PROJECT_ROOT,
    create_project_zip,
    get_or_build_generated_project_zip,
    get_or_build_project_zip,
)

_DIST_DIR = PROJECT_ROOT / "dist"


def _clear_dist() -> None:
    if _DIST_DIR.exists():
        for entry in _DIST_DIR.iterdir():
            if entry.is_file():
                entry.unlink()


@pytest.fixture(autouse=True)
def _clean_packaging_state():
    tools._ZIP_CACHE.update(signature=None, bytes=None, filename=None, report=None)
    tools._GENERATED_ZIP_CACHE.clear()
    _clear_dist()
    yield
    tools._ZIP_CACHE.update(signature=None, bytes=None, filename=None, report=None)
    tools._GENERATED_ZIP_CACHE.clear()
    _clear_dist()


@pytest.fixture
def scratch_generated_project():
    """A small, disposable generated-project tree under
    generated_projects/, removed after the test regardless of outcome -
    stands in for what New Application Generation would actually produce."""
    root = PROJECT_ROOT / "generated_projects" / "_scratch_app"
    (root / "tests").mkdir(parents=True)
    (root / "app.py").write_text("def main():\n    return 'hello'\n")
    (root / "requirements.txt").write_text("streamlit\n")
    (root / "tests" / "test_app.py").write_text(
        "from app import main\n\n\ndef test_main():\n    assert main() == 'hello'\n"
    )
    yield root
    import shutil

    if root.exists():
        shutil.rmtree(root)


@pytest.fixture
def scratch_secret_file():
    """A disposable, innocuously-named project file (outside tests/, since
    content-level secret scanning deliberately skips that directory - see
    tools._scan_project_files) containing something that looks like a real
    secret value. The name deliberately avoids "secret"/"credential"/
    "password" so a test using this fixture actually exercises the
    CONTENT-level check rather than being excluded by name first. Removed
    after the test regardless of outcome."""
    rel_path = "_packaging_scratch_config_values.py"
    abs_path = PROJECT_ROOT / rel_path
    yield rel_path, abs_path
    if abs_path.exists():
        abs_path.unlink()


@pytest.fixture
def scratch_zip_file():
    """A disposable pre-existing .zip file at the project root - removed
    after the test regardless of outcome."""
    abs_path = PROJECT_ROOT / "_packaging_scratch_previous.zip"
    yield abs_path
    if abs_path.exists():
        abs_path.unlink()


def _open_zip(zip_bytes: bytes) -> zipfile.ZipFile:
    import io

    return zipfile.ZipFile(io.BytesIO(zip_bytes))


# ---------------------------------------------------------------------------
# 1-3: basic creation, expected files, directory structure preserved
# ---------------------------------------------------------------------------


def test_basic_zip_creation_returns_nonempty_bytes():
    package = get_or_build_project_zip()
    assert isinstance(package["bytes"], bytes)
    assert len(package["bytes"]) > 0
    assert package["included"] > 0


def test_zip_contains_expected_real_project_files():
    package = get_or_build_project_zip()
    assert "tools.py" in package["files"]
    assert "agent.py" in package["files"]
    assert "requirements.txt" in package["files"]
    assert "README.md" in package["files"]


def test_zip_preserves_directory_structure():
    package = get_or_build_project_zip()
    with _open_zip(package["bytes"]) as zf:
        names = zf.namelist()
    assert "tests/test_tools.py" in names
    assert any(name.startswith("tests/") for name in names)
    # Forward slashes only - never a backslash-style Windows path in the archive.
    assert all("\\" not in name for name in names)


def test_zip_can_be_opened_and_extracted(tmp_path):
    package = get_or_build_project_zip()
    with _open_zip(package["bytes"]) as zf:
        bad_file = zf.testzip()
        assert bad_file is None
        zf.extractall(tmp_path)
    assert (tmp_path / "tools.py").exists()
    assert (tmp_path / "tests" / "test_tools.py").exists()


# ---------------------------------------------------------------------------
# 4-9: security exclusions
# ---------------------------------------------------------------------------


def test_env_file_is_excluded():
    package = get_or_build_project_zip()
    assert ".env" not in package["files"]


def test_env_example_is_included():
    # The real project ships a tracked .env.example - see .gitignore's
    # explicit carve-out for it.
    assert (PROJECT_ROOT / ".env.example").exists()
    package = get_or_build_project_zip()
    assert ".env.example" in package["files"]


def test_api_key_looking_content_is_excluded_even_with_an_innocuous_name(
    scratch_secret_file,
):
    rel_path, abs_path = scratch_secret_file
    abs_path.write_text('GOOGLE_API_KEY = "AIzaSyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"\n')
    package = get_or_build_project_zip(force=True)
    assert rel_path not in package["files"]
    assert package["excluded_secrets"] >= 1


def test_generic_key_value_secret_is_excluded(scratch_secret_file):
    rel_path, abs_path = scratch_secret_file
    abs_path.write_text('SECRET_TOKEN = "some-long-secret-value-12345"\n')
    package = get_or_build_project_zip(force=True)
    assert rel_path not in package["files"]
    assert package["excluded_secrets"] >= 1


def test_credential_named_file_is_excluded_by_name(tmp_path):
    scratch = PROJECT_ROOT / "tests" / "_packaging_scratch_credentials.json"
    scratch.write_text('{"note": "not a real secret value"}')
    try:
        package = get_or_build_project_zip(force=True)
        assert "tests/_packaging_scratch_credentials.json" not in package["files"]
    finally:
        scratch.unlink()


def test_password_named_file_is_excluded_by_name():
    scratch = PROJECT_ROOT / "tests" / "_packaging_scratch_password.txt"
    scratch.write_text("not a real secret value")
    try:
        package = get_or_build_project_zip(force=True)
        assert "tests/_packaging_scratch_password.txt" not in package["files"]
    finally:
        scratch.unlink()


def test_git_directory_is_excluded():
    package = get_or_build_project_zip()
    assert all(not f.startswith(".git/") for f in package["files"])
    assert ".git" not in package["files"]


def test_venv_directory_is_excluded():
    package = get_or_build_project_zip()
    assert all(not f.startswith("venv/") for f in package["files"])
    assert all(not f.startswith(".venv/") for f in package["files"])


def test_pycache_is_excluded():
    package = get_or_build_project_zip()
    assert all("__pycache__" not in f for f in package["files"])


def test_existing_zip_files_are_excluded(scratch_zip_file):
    scratch_zip_file.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty zip
    package = get_or_build_project_zip(force=True)
    assert scratch_zip_file.name not in package["files"]
    assert all(not f.endswith(".zip") for f in package["files"])


def test_test_suite_fixtures_with_fake_keys_are_not_excluded():
    """Regression test: tests/test_logger.py legitimately contains
    fake-but-correctly-formatted example keys/tokens to verify
    logger.sanitize()'s own redaction patterns actually work. A content-level
    secret scan that didn't exempt tests/ would silently exclude this real,
    load-bearing test file (and others like it) from every single build."""
    assert "AIzaSyD-fake" in (PROJECT_ROOT / "tests" / "test_logger.py").read_text(
        encoding="utf-8"
    )
    package = get_or_build_project_zip()
    assert "tests/test_logger.py" in package["files"]


def test_streamlit_secrets_toml_would_be_excluded_by_name():
    # No real .streamlit/secrets.toml exists in this project (see
    # .gitignore), but the exclusion rule must still recognize it by name -
    # verified directly against the same check the scan uses.
    from tools import _is_blocked_file

    assert _is_blocked_file(PROJECT_ROOT / ".streamlit" / "secrets.toml")


# ---------------------------------------------------------------------------
# 10-13: project-root / path safety
# ---------------------------------------------------------------------------


def test_dist_output_folder_itself_is_excluded_from_its_own_archive():
    get_or_build_project_zip()  # first build creates dist/<file>.zip
    package = get_or_build_project_zip(force=True)  # rescans, including dist/
    assert all(not f.startswith("dist/") for f in package["files"])


def test_files_outside_project_root_cannot_be_included(tmp_path):
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("x = 1\n")

    zip_bytes = tools._build_zip_bytes([outside_file])
    with _open_zip(zip_bytes) as zf:
        assert zf.namelist() == []


def test_scan_never_descends_into_excluded_directories(monkeypatch):
    # If _scan_project_files ever stopped pruning noise directories at the
    # directory level, this project's real ~7000-file venv/ would dominate
    # both the included and excluded counts - assert it does not.
    package = get_or_build_project_zip()
    assert package["excluded"] < 200


def test_get_or_build_project_zip_raises_on_empty_project(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_scan_project_files",
        lambda root: {"included": [], "excluded": [], "secret_hits": []},
    )
    with pytest.raises(OSError):
        get_or_build_project_zip(force=True)


# ---------------------------------------------------------------------------
# 14 (partial - tool-level): create_project_zip tool
# ---------------------------------------------------------------------------


def test_create_project_zip_tool_reports_success_and_writes_dist_file():
    result = create_project_zip.invoke({"project_name": "demo project"})
    assert "project package ready" in result.lower()
    assert "demo_project.zip" in result
    assert (PROJECT_ROOT / "dist" / "demo_project.zip").exists()


def test_create_project_zip_tool_default_name_is_timestamp_based():
    result = create_project_zip.invoke({})
    assert "ai_developer_project_" in result


def test_create_project_zip_never_reports_success_without_calling_the_real_builder(
    monkeypatch,
):
    def _boom(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tools, "get_or_build_project_zip", _boom)
    result = create_project_zip.invoke({})
    assert "error" in result.lower()
    assert "project package ready" not in result.lower()


# ---------------------------------------------------------------------------
# 16-18: idempotency
# ---------------------------------------------------------------------------


def test_repeated_calls_reuse_the_cached_archive_without_rebuilding():
    first = get_or_build_project_zip()
    assert first["rebuilt"] is True

    second = get_or_build_project_zip()
    assert second["rebuilt"] is False
    assert second["bytes"] == first["bytes"]


def test_renaming_alone_does_not_force_a_rebuild():
    first = get_or_build_project_zip(project_name="alpha")
    second = get_or_build_project_zip(project_name="bravo")
    assert second["rebuilt"] is False
    assert second["filename"] == "bravo.zip"
    assert second["bytes"] == first["bytes"]


def test_project_change_triggers_a_real_rebuild():
    rel_path = "tests/_packaging_scratch_change.py"
    abs_path = PROJECT_ROOT / rel_path
    try:
        first = get_or_build_project_zip()
        assert rel_path not in first["files"]

        abs_path.write_text("NEW_VALUE = 123\n")
        second = get_or_build_project_zip()

        assert second["rebuilt"] is True
        assert rel_path in second["files"]
    finally:
        if abs_path.exists():
            abs_path.unlink()


def test_force_rebuild_always_rebuilds_even_when_unchanged():
    get_or_build_project_zip()
    forced = get_or_build_project_zip(force=True)
    assert forced["rebuilt"] is True


# ---------------------------------------------------------------------------
# get_or_build_generated_project_zip: scoped packaging for one generated
# sub-project (New Application Generation's own ZIP delivery, e.g.
# "hrms.zip") - a separate archive from the whole-assistant-project one
# above, containing ONLY that project's own files.
# ---------------------------------------------------------------------------


def test_generated_project_zip_contains_only_that_projects_files(
    scratch_generated_project,
):
    package = get_or_build_generated_project_zip(
        source_dir="generated_projects/_scratch_app"
    )
    assert "_scratch_app/app.py" in package["files"]
    assert "_scratch_app/requirements.txt" in package["files"]
    assert "_scratch_app/tests/test_app.py" in package["files"]
    # Never any file from this whole assistant project - only the generated
    # sub-project's own three files above.
    assert package["included"] == 3
    assert all(f.startswith("_scratch_app/") for f in package["files"])


def test_generated_project_zip_top_level_entry_matches_folder_name(
    scratch_generated_project,
):
    package = get_or_build_generated_project_zip(
        source_dir="generated_projects/_scratch_app"
    )
    with _open_zip(package["bytes"]) as zf:
        names = zf.namelist()
    assert "_scratch_app/app.py" in names
    assert not any(n.startswith("generated_projects/") for n in names)


def test_generated_project_zip_defaults_filename_to_folder_name(
    scratch_generated_project,
):
    package = get_or_build_generated_project_zip(
        source_dir="generated_projects/_scratch_app"
    )
    assert package["filename"] == "scratch_app.zip"


def test_generated_project_zip_honors_explicit_project_name(
    scratch_generated_project,
):
    package = get_or_build_generated_project_zip(
        source_dir="generated_projects/_scratch_app", project_name="HRMS Demo"
    )
    assert package["filename"] == "hrms_demo.zip"


def test_generated_project_zip_raises_for_missing_source_dir():
    with pytest.raises(OSError):
        get_or_build_generated_project_zip(
            source_dir="generated_projects/does_not_exist"
        )


def test_generated_project_zip_rejects_path_traversal():
    with pytest.raises(OSError):
        get_or_build_generated_project_zip(source_dir="../outside")


def test_generated_project_zip_is_idempotent(scratch_generated_project):
    first = get_or_build_generated_project_zip(
        source_dir="generated_projects/_scratch_app"
    )
    assert first["rebuilt"] is True
    second = get_or_build_generated_project_zip(
        source_dir="generated_projects/_scratch_app"
    )
    assert second["rebuilt"] is False
    assert second["bytes"] == first["bytes"]


def test_generated_project_zip_rebuilds_when_that_project_changes(
    scratch_generated_project,
):
    get_or_build_generated_project_zip(source_dir="generated_projects/_scratch_app")
    (scratch_generated_project / "new_file.py").write_text("x = 1\n")
    second = get_or_build_generated_project_zip(
        source_dir="generated_projects/_scratch_app"
    )
    assert second["rebuilt"] is True
    assert "_scratch_app/new_file.py" in second["files"]


def test_generated_project_zip_cache_is_independent_of_whole_project_cache(
    scratch_generated_project,
):
    """Building the whole-assistant-project archive and a generated
    sub-project's archive must never interfere with each other's cache."""
    whole = get_or_build_project_zip()
    generated = get_or_build_generated_project_zip(
        source_dir="generated_projects/_scratch_app"
    )
    assert whole["filename"] != generated["filename"]
    assert whole["bytes"] != generated["bytes"]
    assert "_scratch_app/app.py" not in whole["files"]


def test_create_project_zip_tool_with_source_dir_packages_generated_project(
    scratch_generated_project,
):
    result = create_project_zip.invoke(
        {"source_dir": "generated_projects/_scratch_app"}
    )
    assert "project package ready" in result.lower()
    assert "scratch_app.zip" in result
    assert (PROJECT_ROOT / "dist" / "scratch_app.zip").exists()
