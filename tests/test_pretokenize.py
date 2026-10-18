"""Tests for the French pre-tokenization rules.

Each expected list is the chunking I want, written by hand from the rules in
src/frbpe/pretokenize.py, before running the code.
"""

import random

import pytest

from frbpe.pretokenize import PATTERNS, PreTokenizer

NBSP = " "
NNBSP = " "


def check(got, expected):
    assert got == expected, f"ECHEC: obtenu {got!r}, attendu {expected!r}"


@pytest.mark.parametrize(
    "text, expected",
    [
        # The elided word keeps its apostrophe, the next word starts clean.
        ("L'école d'été", ["L'", "école", " d'", "été"]),
        ("L’été", ["L’", "été"]),                      # typographic ’
        ("jusqu'à", ["jusqu'", "à"]),                           # long form first
        ("lorsqu'il", ["lorsqu'", "il"]),
        ("aujourd'hui", ["aujourd'hui"]),                        # one word
        (" aujourd’hui", [" aujourd’hui"]),
        # A quote is not an elision: no letter list match before it.
        ("dit 'non'", ["dit", " '", "non", "'"]),
        # Not in the closed list: handled as in GPT-4.
        ("O'Neill", ["O", "'Neill"]),
    ],
)
def test_fr_elision(text, expected):
    check(PreTokenizer("fr_elision").split(text), expected)


@pytest.mark.parametrize(
    "text, expected",
    [
        # GPT-4 gives "1", " ", "000", " ", "000": 5 chunks. French rule: 3.
        ("1 000 000", ["1", " 000", " 000"]),
        (f"12{NBSP}345", ["12", f"{NBSP}345"]),
        ("le 14", ["le", " 14"]),
        # Groups of at most 3 digits, as in GPT-4.
        ("1789", ["178", "9"]),
    ],
)
def test_fr_number(text, expected):
    check(PreTokenizer("fr_number").split(text), expected)


@pytest.mark.parametrize(
    "text, expected",
    [
        (f"Prix{NBSP}: oui", ["Prix", f"{NBSP}:", " oui"]),
        (f"vrai{NNBSP}!", ["vrai", f"{NNBSP}!"]),
        ("a ; b", ["a", " ;", " b"]),
    ],
)
def test_fr_punctuation(text, expected):
    check(PreTokenizer("fr_punct").split(text), expected)


def test_gpt4_leaves_no_break_space_alone():
    # Reference behaviour that the French rule changes: 4 chunks instead of 3.
    check(PreTokenizer("gpt4").split(f"Prix{NBSP}: oui"), ["Prix", NBSP, ":", " oui"])


def test_full_fr_pattern_combines_all_rules():
    text = f"Qu'il paie 1 000{NBSP}€ aujourd'hui{NBSP}!"
    expected = ["Qu'", "il", " paie", " 1", " 000", f"{NBSP}€", " aujourd'hui", f"{NBSP}!"]
    check(PreTokenizer("fr").split(text), expected)


@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_every_pattern_is_lossless(name):
    """BPE can only rebuild the text if the chunks cover it entirely, with no
    character dropped. Checked on French text and on random code points."""
    pre = PreTokenizer(name)
    rng = random.Random(99)
    samples = [
        f"L’été, jusqu'à 1 000{NBSP}€{NNBSP}! « Oui »\n\n  fin  ",
        "été \t\r\n",
    ]
    for _ in range(300):
        cps = [rng.randint(0, 0x10FFFF) for _ in range(rng.randint(0, 30))]
        samples.append("".join(chr(c) if not 0xD800 <= c <= 0xDFFF else " " for c in cps))
    for text in samples:
        check("".join(pre.split(text)), text)
