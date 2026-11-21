"""Helpers for putting untrusted retrieved text into prompts."""

import re

_TAG = re.compile(r"</?\s*document\s*>", re.IGNORECASE)


def escape_document_text(text: str) -> str:
    """Neutralise ``<document>`` / ``</document>`` inside retrieved text.

    Retrieved chunks are wrapped in ``<document>`` blocks so the model can tell data from
    instructions; a PDF containing a literal ``</document>`` could otherwise "close" the block
    and smuggle text out of it.
    """
    return _TAG.sub(lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"), text or "")
