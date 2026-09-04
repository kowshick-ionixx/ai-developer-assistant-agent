"""
Tests for documents.py.

Covers upload validation (type/size/name), text/PDF/DOCX parsing, corrupted
and empty files, invalid encoding, secret redaction, and the document context
block sent to the agent - all pure Python, no Streamlit runtime and no
network/API calls required.
"""

import io

import docx
import pytest
from pypdf import PdfWriter

from documents import (
    MAX_UPLOAD_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
    build_document_context_block,
    format_size,
    process_upload,
    validate_upload,
)


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _make_pdf_bytes(num_pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# validate_upload: file type / size / name
# ---------------------------------------------------------------------------


def test_validate_upload_accepts_supported_type():
    assert validate_upload("example.py", 100) is None


def test_validate_upload_rejects_unsupported_type():
    error = validate_upload("malicious.exe", 100)
    assert error is not None
    assert "unsupported" in error.lower()


def test_validate_upload_rejects_oversized_file():
    error = validate_upload("big.txt", MAX_UPLOAD_SIZE_BYTES + 1)
    assert error is not None
    assert "maximum allowed size" in error.lower()


def test_validate_upload_accepts_file_at_exact_size_limit():
    assert validate_upload("exact.txt", MAX_UPLOAD_SIZE_BYTES) is None


def test_validate_upload_rejects_empty_file():
    error = validate_upload("empty.txt", 0)
    assert error is not None
    assert "empty" in error.lower()


def test_supported_extensions_match_the_required_list():
    assert SUPPORTED_EXTENSIONS == {
        ".py",
        ".txt",
        ".md",
        ".json",
        ".csv",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".pdf",
        ".docx",
    }


# ---------------------------------------------------------------------------
# process_upload: path/filename security
# ---------------------------------------------------------------------------


def test_process_upload_reduces_path_traversal_filename_to_base_name():
    result = process_upload("../../secret.txt", b"hello world")
    assert "error" not in result
    assert result["filename"] == "secret.txt"


def test_process_upload_reduces_windows_style_path_to_base_name():
    result = process_upload("C:\\Windows\\System32\\evil.py", b"print(1)")
    assert "error" not in result
    assert result["filename"] == "evil.py"


# ---------------------------------------------------------------------------
# process_upload: text formats
# ---------------------------------------------------------------------------


def test_process_upload_parses_python_file():
    result = process_upload("example.py", b"def add(a, b):\n    return a + b\n")
    assert result["file_type"] == "Python source"
    assert "def add" in result["content"]
    assert result["truncated"] is False


def test_process_upload_rejects_invalid_encoding():
    result = process_upload("bad.txt", b"\xff\xfe\x00\x01binarydata")
    assert "error" in result
    assert "utf-8" in result["error"].lower()


def test_process_upload_truncates_oversized_content():
    huge_content = ("x" * 20000).encode("utf-8")
    result = process_upload("huge.txt", huge_content)
    assert result["truncated"] is True
    assert result["content"].endswith("(truncated)")


# ---------------------------------------------------------------------------
# process_upload: secret redaction (defense in depth for uploaded content)
# ---------------------------------------------------------------------------


def test_process_upload_redacts_gemini_api_key():
    raw = b"GEMINI_API_KEY=super-secret-value-123\nother=fine\n"
    result = process_upload("config.txt", raw)
    assert "super-secret-value-123" not in result["content"]
    assert "[REDACTED]" in result["content"]


def test_process_upload_redacts_password_and_token():
    raw = b"PASSWORD=hunter2\nTOKEN=abcdef123456\n"
    result = process_upload("secrets.txt", raw)
    assert "hunter2" not in result["content"]
    assert "abcdef123456" not in result["content"]


# ---------------------------------------------------------------------------
# process_upload: PDF
# ---------------------------------------------------------------------------


def test_process_upload_pdf_with_no_extractable_text_reports_clear_error():
    # A validly-structured PDF with blank pages has no extractable text -
    # this must be reported clearly, never crash and never fabricate content.
    result = process_upload("blank.pdf", _make_pdf_bytes())
    assert "error" in result
    assert "no readable text" in result["error"].lower()


def test_process_upload_rejects_corrupted_pdf():
    result = process_upload("corrupted.pdf", b"%PDF-1.4 this is not a real pdf")
    assert "error" in result
    assert (
        "corrupted" in result["error"].lower()
        or "could not be read" in result["error"].lower()
    )


# ---------------------------------------------------------------------------
# process_upload: DOCX
# ---------------------------------------------------------------------------


def test_process_upload_parses_docx_text():
    raw = _make_docx_bytes(["Hello from a Word document.", "Second paragraph."])
    result = process_upload("notes.docx", raw)
    assert "error" not in result
    assert result["file_type"] == "Word document"
    assert "Hello from a Word document." in result["content"]


def test_process_upload_docx_redacts_secrets():
    raw = _make_docx_bytes(["TOKEN=verysecrettoken123456"])
    result = process_upload("notes.docx", raw)
    assert "verysecrettoken123456" not in result["content"]


def test_process_upload_rejects_corrupted_docx():
    result = process_upload("corrupted.docx", b"not a real docx file at all")
    assert "error" in result
    assert (
        "corrupted" in result["error"].lower()
        or "could not be read" in result["error"].lower()
    )


def test_process_upload_docx_with_no_text_reports_clear_error():
    raw = _make_docx_bytes([])
    result = process_upload("empty.docx", raw)
    assert "error" in result
    assert "no readable text" in result["error"].lower()


# ---------------------------------------------------------------------------
# format_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size_bytes,expected_unit",
    [(500, "B"), (4300, "KB"), (5 * 1024 * 1024, "MB")],
)
def test_format_size_uses_expected_unit(size_bytes, expected_unit):
    assert format_size(size_bytes).endswith(expected_unit)


# ---------------------------------------------------------------------------
# build_document_context_block
# ---------------------------------------------------------------------------


def test_build_document_context_block_empty_when_no_files():
    assert build_document_context_block([]) == ""


def test_build_document_context_block_labels_content_as_untrusted():
    files = [{"filename": "a.py", "file_type": "Python source", "content": "print(1)"}]
    block = build_document_context_block(files)
    assert "untrusted" in block.lower()
    assert "a.py" in block
    assert "print(1)" in block
