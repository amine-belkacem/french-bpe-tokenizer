"""Pre-tokenization: cut the text into chunks before BPE sees it.

BPE never merges across two chunks. The regex therefore decides which tokens
can exist at all: with GPT-4's pattern, " maison" can become one token, but
"maison." cannot, because the dot always starts a new chunk.

Every pattern here is built from the same five pieces, so that each French
rule can be switched on alone and its effect measured on its own:

    contraction | word | number | punctuation | whitespace

GPT-4 uses the original pieces. The French pieces replace one of them:

* elision      " l'", " qu'", " jusqu'" become their own chunk, followed by
               the word without its apostrophe ("école" instead of "'école").
               Replaces the English contraction branch ("'s", "'ll", ...).
* number       a number may carry the space before it, including the no-break
               spaces used as thousands separator: "1 000" -> "1", " 000"
               instead of "1", " ", "000".
* punctuation  French puts a (narrow) no-break space before ; : ! ? and after
               «. GPT-4 only allows a plain space before punctuation, so the
               no-break space ended up alone in its own chunk.

Hyphens are left as GPT-4 handles them ("peut", "-être"). Gluing compounds
into one chunk would create many rare chunks; I did not try it.
"""

from __future__ import annotations

import regex

# Pieces copied from tiktoken 0.14.0, tiktoken_ext/openai_public.py, cl100k_base.
GPT4_CONTRACTION = r"'(?i:[sdmt]|ll|ve|re)"
GPT4_WORD = r"[^\r\n\p{L}\p{N}]?+\p{L}++"
GPT4_NUMBER = r"\p{N}{1,3}+"
GPT4_PUNCTUATION = r" ?[^\s\p{L}\p{N}]++[\r\n]*+"
GPT4_WHITESPACE = r"\s++$|\s*[\r\n]|\s+(?!\S)|\s"

_APOSTROPHE = "['’]"           # straight ' and typographic ’
_SPACES = "[   ]"          # space, no-break space, narrow no-break space

# "aujourd'hui" is a single word, not an elision: keep it whole. Elided words
# are a closed list in French; longer forms come first so "jusqu'" is not read
# as "j" + "usqu'". The lookahead requires a letter after the apostrophe, so a
# closing quote ("il dit 'non'") is not taken for an elision.
FR_ELISION = (
    rf"[^\r\n\p{{L}}\p{{N}}]?+(?i:aujourd{_APOSTROPHE}hui)\b"
    rf"|[^\r\n\p{{L}}\p{{N}}]?+(?i:jusqu|lorsqu|puisqu|quoiqu|presqu|quelqu|qu|[cdjlmnst])"
    rf"{_APOSTROPHE}(?=\p{{L}})"
)
FR_NUMBER = rf"{_SPACES}?\p{{N}}{{1,3}}+"
FR_PUNCTUATION = rf"{_SPACES}?[^\s\p{{L}}\p{{N}}]++[\r\n]*+"


def _join(*pieces: str) -> str:
    return "|".join(pieces)


PATTERNS = {
    "gpt4": _join(GPT4_CONTRACTION, GPT4_WORD, GPT4_NUMBER, GPT4_PUNCTUATION, GPT4_WHITESPACE),
    # One French rule at a time, everything else as in GPT-4.
    "fr_elision": _join(FR_ELISION, GPT4_WORD, GPT4_NUMBER, GPT4_PUNCTUATION, GPT4_WHITESPACE),
    "fr_number": _join(GPT4_CONTRACTION, GPT4_WORD, FR_NUMBER, GPT4_PUNCTUATION, GPT4_WHITESPACE),
    "fr_punct": _join(GPT4_CONTRACTION, GPT4_WORD, GPT4_NUMBER, FR_PUNCTUATION, GPT4_WHITESPACE),
    # All French rules together.
    "fr": _join(FR_ELISION, GPT4_WORD, FR_NUMBER, FR_PUNCTUATION, GPT4_WHITESPACE),
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
