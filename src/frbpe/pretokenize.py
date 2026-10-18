"""Pre-tokenization: cut the text into chunks before BPE sees it.

BPE never merges across two chunks. The regex therefore decides which tokens
can exist at all: with GPT-4's pattern, " maison" can become one token, but
"maison." cannot, because the dot always starts a new chunk.

This module holds the patterns compared in the project. Only the GPT-4 one is
here for now; the French variant comes in milestone J2.
"""

from __future__ import annotations

import regex

# Copied from tiktoken 0.14.0, tiktoken_ext/openai_public.py, cl100k_base.
# The "'(?i:[sdmt]|ll|ve|re)" branch is written for English contractions
# ("it's", "we'll"). French elisions such as "l'" or "qu'" are not covered:
# the apostrophe ends up glued to the *next* word ("'école").
GPT4_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)

PATTERNS = {
    "gpt4": GPT4_PATTERN,
}


class PreTokenizer:
    """Compiled pattern with a name, so a saved model can record which one it used."""

    def __init__(self, name: str) -> None:
        if name not in PATTERNS:
            known = ", ".join(sorted(PATTERNS))
            raise ValueError(f"unknown pattern {name!r}, expected one of: {known}")
        self.name = name
        self._regex = regex.compile(PATTERNS[name])

    def split(self, text: str) -> list[str]:
        """Return the chunks. Joining them gives back ``text`` exactly."""
        return self._regex.findall(text)
