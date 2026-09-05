"""Tests for string_utils.py."""

import pytest

from string_utils import slugify


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("Hello World", "hello-world"),
        ("Python_Programming 101!", "python-programming-101"),
        ("  Multiple   Spaces__and--Punctuation?! ", "multiple-spaces-and-punctuation"),
        ("---Already-Slugged---", "already-slugged"),
        ("", ""),
        ("12345", "12345"),
    ],
)
def test_slugify(input_text, expected):
    assert slugify(input_text) == expected
