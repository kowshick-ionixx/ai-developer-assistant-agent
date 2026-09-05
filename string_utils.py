"""String utility functions."""

import re


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug.

    - Convert to lowercase
    - Replace spaces and underscores with hyphens
    - Remove punctuation
    - Collapse multiple hyphens and strip edges
    """
    if not isinstance(text, str):
        text = str(text)
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep alphanumeric, whitespace, underscores, hyphens)
    text = re.sub(r"[^\w\s-]", "", text)
    # Replace spaces and underscores with hyphens
    text = re.sub(r"[\s_]+", "-", text)
    # Collapse multiple hyphens into a single hyphen
    text = re.sub(r"-+", "-", text)
    # Strip leading/trailing hyphens
    return text.strip("-")
