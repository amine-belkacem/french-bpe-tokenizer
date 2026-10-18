"""Text perturbations for the robustness experiment (milestone 6).

Each one is something real French text goes through: typed without accents
on a QWERTY keyboard, shouted in capitals, pasted from a word processor that
"curls" apostrophes, saved by a macOS tool in NFD, or cleaned by a script
that drops no-break spaces. The question is how many extra tokens each
tokenizer pays for the same content.
"""

from __future__ import annotations

import unicodedata
from typing import Callable

import regex

_COMBINING = regex.compile(r"\p{M}")
_NO_BREAK_SPACES = str.maketrans({" ": " ", " ": " "})


def strip_accents(text: str) -> str:
    """"école" -> "ecole". Decompose, drop the combining marks, recompose.
    Letters with no decomposition ("œ", "ø") are kept as they are."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", _COMBINING.sub("", decomposed))


def to_upper(text: str) -> str:
    return text.upper()


def curly_apostrophes(text: str) -> str:
    """Every straight apostrophe becomes the typographic one."""
    return text.replace("'", "’")


def to_nfd(text: str) -> str:
    """"é" (1 code point) becomes "e" + U+0301 (2 code points)."""
    return unicodedata.normalize("NFD", text)


def plain_spaces(text: str) -> str:
    """No-break and narrow no-break spaces become ordinary spaces."""
    return text.translate(_NO_BREAK_SPACES)


PERTURBATIONS: dict[str, Callable[[str], str]] = {
    "no accents": strip_accents,
    "upper case": to_upper,
    "curly apostrophes": curly_apostrophes,
    "NFD": to_nfd,
    "no no-break spaces": plain_spaces,
}
