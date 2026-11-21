"""Helpers for working with arXiv identifiers."""

import re

# Matches a trailing version suffix such as "v1", "v12".
_VERSION_SUFFIX = re.compile(r"v\d+$")


def strip_arxiv_version(arxiv_id: str) -> str:
    """Remove the trailing version suffix from an arXiv id.

    The previous inline logic (``arxiv_id.split("v")[0]``) cut the id at the first "v",
    which corrupts old-style ids whose archive name contains a "v"
    (e.g. ``solv-int/9901001v1`` -> ``sol``).

    >>> strip_arxiv_version("2401.12345v2")
    '2401.12345'
    >>> strip_arxiv_version("solv-int/9901001v1")
    'solv-int/9901001'
    """
    return _VERSION_SUFFIX.sub("", arxiv_id.strip())


def arxiv_pdf_url(arxiv_id: str) -> str:
    """Canonical (version-less) PDF URL for an arXiv id."""
    return f"https://arxiv.org/pdf/{strip_arxiv_version(arxiv_id)}.pdf"


def arxiv_abs_url(arxiv_id: str) -> str:
    """Canonical (version-less) abstract page URL for an arXiv id."""
    return f"https://arxiv.org/abs/{strip_arxiv_version(arxiv_id)}"
