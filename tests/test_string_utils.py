"""Tests for string_utils.py."""

import pytest

from string_utils import is_palindrome, slugify


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


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("racecar", True),
        ("A man, a plan, a canal: Panama", True),
        ("hello", False),
        (12321, True),
        ("", True),
        ("No 'x' in Nixon", True),
    ],
)
def test_is_palindrome(input_text, expected):
    assert is_palindrome(input_text) == expected
