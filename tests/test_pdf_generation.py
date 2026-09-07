"""
Tests for tools.py's PDF generation feature: the create_pdf tool and its
shared implementation (_build_pdf_bytes, _pdf_safe_filename, _unique_pdf_path,
get_generated_pdf_path).

These exercise the real PDF-generation logic directly against this project's
actual filesystem (the same way tests/test_packaging.py exercises
create_project_zip) - the real "generated_files/" output folder this feature
writes into is cleared before and after every test in this file so no test
can see another test's leftover PDF.
"""

import re

import pytest
from pypdf import PdfReader

import tools
from tools import PROJECT_ROOT, create_pdf, get_generated_pdf_path

_OUTPUT_DIR = PROJECT_ROOT / "generated_files"

_SAMPLE_CONTENT = """# Introduction
Python is a popular, beginner-friendly programming language.

## Key Concepts
- Variables store data
- Functions group reusable code
- Loops repeat actions

## Example
```python
def greet(name):
    return f"Hello, {name}!"
```

## Summary
Python is great for beginners.
"""


def _clear_output_dir() -> None:
    if _OUTPUT_DIR.exists():
        for entry in _OUTPUT_DIR.iterdir():
            if entry.is_file():
                entry.unlink()


@pytest.fixture(autouse=True)
def _clean_pdf_output():
    _clear_output_dir()
    yield
    _clear_output_dir()


def _path_from_result(result: str) -> str:
    match = re.search(r"^Path:\s*(\S+)\s*$", result, flags=re.MULTILINE)
    assert match, f"no 'Path: ...' line found in result: {result!r}"
    return match.group(1)


# ---------------------------------------------------------------------------
# 1-3: basic creation, file actually exists, generated file is a valid PDF
# ---------------------------------------------------------------------------


def test_create_pdf_reports_success():
    result = create_pdf.invoke({"title": "Python Basics", "content": _SAMPLE_CONTENT})
    assert "pdf created successfully" in result.lower()


def test_create_pdf_file_actually_exists_on_disk():
    result = create_pdf.invoke({"title": "Python Basics", "content": _SAMPLE_CONTENT})
    rel_path = _path_from_result(result)
    full_path = PROJECT_ROOT / rel_path
    assert full_path.is_file()
    assert full_path.stat().st_size > 0
    assert full_path.parent == _OUTPUT_DIR


def test_generated_file_is_a_valid_pdf():
    result = create_pdf.invoke({"title": "Python Basics", "content": _SAMPLE_CONTENT})
    rel_path = _path_from_result(result)
    full_path = PROJECT_ROOT / rel_path

    raw = full_path.read_bytes()
    assert raw.startswith(b"%PDF")

    reader = PdfReader(str(full_path))  # raises PdfReadError if corrupted
    assert len(reader.pages) >= 1
    text = reader.pages[0].extract_text() or ""
    assert "Python Basics" in text
    assert "Introduction" in text


def test_long_multi_section_document_generates_a_multi_page_pdf():
    """Long documents (e.g. a full application-documentation PDF spanning
    many real sections) must actually flow across multiple pages, not
    truncate or crash."""
    sections = []
    for i in range(1, 31):
        sections.append(f"## Section {i}\n" + ("Detailed paragraph text. " * 40))
    long_content = "\n\n".join(sections)

    result = create_pdf.invoke({"title": "Long Document", "content": long_content})
    rel_path = _path_from_result(result)
    full_path = PROJECT_ROOT / rel_path

    reader = PdfReader(str(full_path))
    assert len(reader.pages) > 1
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Section 1" in full_text
    assert "Section 30" in full_text


def test_generated_pdf_includes_code_block_content():
    result = create_pdf.invoke({"title": "Python Basics", "content": _SAMPLE_CONTENT})
    rel_path = _path_from_result(result)
    full_path = PROJECT_ROOT / rel_path
    reader = PdfReader(str(full_path))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "def greet" in full_text


# ---------------------------------------------------------------------------
# 4: empty content is handled
# ---------------------------------------------------------------------------


def test_empty_content_is_rejected_with_a_clear_error():
    result = create_pdf.invoke({"title": "Python Basics", "content": ""})
    assert "error" in result.lower()
    assert "pdf created successfully" not in result.lower()
    assert not any(_OUTPUT_DIR.glob("*.pdf"))


def test_whitespace_only_content_is_rejected():
    result = create_pdf.invoke({"title": "Python Basics", "content": "   \n\n   "})
    assert "error" in result.lower()


def test_empty_title_is_rejected():
    result = create_pdf.invoke({"title": "", "content": _SAMPLE_CONTENT})
    assert "error" in result.lower()
    assert not any(_OUTPUT_DIR.glob("*.pdf"))


def test_oversized_content_is_rejected():
    huge_content = "word " * 20_000  # well over _MAX_PDF_CONTENT_CHARS
    result = create_pdf.invoke({"title": "Too Big", "content": huge_content})
    assert "error" in result.lower()
    assert "too long" in result.lower()
    assert not any(_OUTPUT_DIR.glob("*.pdf"))


# ---------------------------------------------------------------------------
# 5: unsafe filename/path is handled safely
# ---------------------------------------------------------------------------


def test_path_traversal_filename_is_sanitized_not_escaped():
    result = create_pdf.invoke(
        {
            "title": "Test",
            "content": _SAMPLE_CONTENT,
            "filename": "../../../etc/passwd",
        }
    )
    rel_path = _path_from_result(result)
    full_path = (PROJECT_ROOT / rel_path).resolve()
    # Must still resolve safely inside generated_files/, never escape it.
    full_path.relative_to(_OUTPUT_DIR)
    assert full_path.is_file()


def test_absolute_windows_path_filename_is_sanitized_not_escaped():
    result = create_pdf.invoke(
        {
            "title": "Test",
            "content": _SAMPLE_CONTENT,
            "filename": "C:\\Windows\\System32\\evil",
        }
    )
    rel_path = _path_from_result(result)
    full_path = (PROJECT_ROOT / rel_path).resolve()
    full_path.relative_to(_OUTPUT_DIR)
    assert full_path.is_file()


def test_filename_with_special_characters_is_sanitized():
    result = create_pdf.invoke(
        {
            "title": "Test",
            "content": _SAMPLE_CONTENT,
            "filename": "  My File!! @#$%^&*() Report  ",
        }
    )
    rel_path = _path_from_result(result)
    filename = rel_path.rsplit("/", 1)[-1]
    assert re.fullmatch(r"[a-z0-9_]+\.pdf", filename)


def test_duplicate_request_never_overwrites_the_earlier_pdf():
    first = create_pdf.invoke({"title": "Python Basics", "content": _SAMPLE_CONTENT})
    first_path = PROJECT_ROOT / _path_from_result(first)
    first_bytes = first_path.read_bytes()

    second = create_pdf.invoke(
        {"title": "Python Basics", "content": _SAMPLE_CONTENT + "\nMore content."}
    )
    second_path = PROJECT_ROOT / _path_from_result(second)

    assert second_path != first_path
    assert first_path.is_file()  # the original file was never overwritten
    assert first_path.read_bytes() == first_bytes


def test_get_generated_pdf_path_rejects_paths_outside_generated_files():
    # Even a syntactically valid in-project path must be rejected if it
    # isn't actually inside generated_files/ - defense in depth beyond the
    # filename sanitization above.
    assert get_generated_pdf_path("tools.py") is None
    assert get_generated_pdf_path("../outside.pdf") is None
    assert get_generated_pdf_path("../requirements.txt") is None


def test_get_generated_pdf_path_returns_real_file_inside_generated_files():
    result = create_pdf.invoke({"title": "Python Basics", "content": _SAMPLE_CONTENT})
    rel_path = _path_from_result(result)
    resolved = get_generated_pdf_path(rel_path)
    assert resolved is not None
    assert resolved.is_file()


def test_get_generated_pdf_path_returns_none_for_nonexistent_file():
    assert get_generated_pdf_path("generated_files/does_not_exist.pdf") is None


# ---------------------------------------------------------------------------
# 6: PDF generation failure is handled
# ---------------------------------------------------------------------------


def test_reportlab_not_installed_is_reported_clearly(monkeypatch):
    def _boom(title, content):
        raise ImportError("No module named 'reportlab'")

    monkeypatch.setattr(tools, "_build_pdf_bytes", _boom)
    result = create_pdf.invoke({"title": "Python Basics", "content": _SAMPLE_CONTENT})
    assert "error" in result.lower()
    assert "reportlab" in result.lower()
    assert not any(_OUTPUT_DIR.glob("*.pdf"))


def test_unexpected_rendering_failure_is_handled_gracefully(monkeypatch):
    def _boom(title, content):
        raise RuntimeError("layout engine exploded")

    monkeypatch.setattr(tools, "_build_pdf_bytes", _boom)
    result = create_pdf.invoke({"title": "Python Basics", "content": _SAMPLE_CONTENT})
    assert result == "Error: PDF generation failed. Please try again."
    assert "layout engine exploded" not in result
    assert not any(_OUTPUT_DIR.glob("*.pdf"))


def test_invalid_generated_bytes_are_never_reported_as_success(monkeypatch):
    monkeypatch.setattr(tools, "_build_pdf_bytes", lambda title, content: b"not a pdf")
    result = create_pdf.invoke({"title": "Python Basics", "content": _SAMPLE_CONTENT})
    assert "error" in result.lower()
    assert not any(_OUTPUT_DIR.glob("*.pdf"))


def test_filesystem_write_error_is_handled_gracefully(monkeypatch):
    from pathlib import Path

    def _boom_write(self, data):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", _boom_write)
    result = create_pdf.invoke({"title": "Python Basics", "content": _SAMPLE_CONTENT})
    assert result == "Error: PDF generation failed. Please try again."


# ---------------------------------------------------------------------------
# Workflow 1 (PDF specification -> application): the mechanical round trip
# a real usage of this feature depends on - a specification PDF create_pdf
# generates can be re-uploaded and its real text extracted via
# documents.process_upload, exactly what the chat attachment pipeline does
# when a user later uploads it and says "build the app from this PDF". The
# LLM's own requirement-analysis reasoning over that extracted text is not
# exercised here (this project's tests never call the live Gemini API) -
# only the deterministic PDF <-> text pipeline both sides of that step
# actually depend on.
# ---------------------------------------------------------------------------


def test_specification_pdf_round_trips_through_upload_extraction():
    import documents

    spec_content = (
        "# Todo Application\n\n"
        "## Technology Stack\nPython + Streamlit + SQLite\n\n"
        "## Features\n- Add Todo\n- Edit Todo\n- Delete Todo\n- Mark Todo complete\n\n"
        "## Database Requirements\ntodos table: id, title, description, completed, "
        "created_at\n"
    )
    result = create_pdf.invoke(
        {"title": "Todo App Project Specification", "content": spec_content}
    )
    rel_path = _path_from_result(result)
    pdf_bytes = (PROJECT_ROOT / rel_path).read_bytes()

    extracted = documents.process_upload("Todo_App_Specification.pdf", pdf_bytes)

    assert "error" not in extracted
    assert "Todo Application" in extracted["content"]
    assert "Add Todo" in extracted["content"]
    assert "SQLite" in extracted["content"]


# ---------------------------------------------------------------------------
# Markdown parsing helper (_parse_pdf_markdown) - the structural building
# block _build_pdf_bytes relies on.
# ---------------------------------------------------------------------------


def test_parse_pdf_markdown_recognizes_headings_bullets_paragraphs_and_code():
    blocks = tools._parse_pdf_markdown(
        "# Title\n"
        "Some intro paragraph text.\n\n"
        "## Section\n"
        "- one\n"
        "- two\n\n"
        "```python\nx = 1\n```\n"
    )
    types = [b[0] for b in blocks]
    assert types == [
        "heading1",
        "paragraph",
        "heading2",
        "bullet",
        "bullet",
        "code",
    ]
    assert blocks[-1][1] == "x = 1"


# ---------------------------------------------------------------------------
# Filename sanitization helper (_pdf_safe_filename)
# ---------------------------------------------------------------------------


def test_pdf_safe_filename_falls_back_to_document_for_empty_input():
    assert tools._pdf_safe_filename("") == "document.pdf"
    assert tools._pdf_safe_filename("!!!") == "document.pdf"


def test_pdf_safe_filename_lowercases_and_replaces_punctuation():
    assert tools._pdf_safe_filename("Python Basics!") == "python_basics.pdf"
