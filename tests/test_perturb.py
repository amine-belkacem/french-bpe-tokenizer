"""Tests for src/frbpe/perturb.py, expected strings written by hand."""

import pytest

from frbpe.perturb import (
    PERTURBATIONS,
    curly_apostrophes,
    plain_spaces,
    strip_accents,
    to_nfd,
    to_upper,
)


def check(got, expected):
    assert got == expected, f"ECHEC: obtenu {got!r}, attendu {expected!r}"


def test_strip_accents():
    check(strip_accents("École d'été, où ça ?"), "Ecole d'ete, ou ca ?")


def test_strip_accents_keeps_letters_without_decomposition():
    # "œ" is its own letter in Unicode, not "o" + a mark: it stays.
    check(strip_accents("cœur"), "cœur")


def test_to_upper():
    check(to_upper("l'été"), "L'ÉTÉ")


def test_curly_apostrophes():
    check(curly_apostrophes("l'école d'été"), "l’école d’été")


def test_to_nfd_splits_accented_letters():
    # "été": 3 code points in NFC, 5 in NFD (e + U+0301, t, e + U+0301).
    check(len("été"), 3)
    check(len(to_nfd("été")), 5)
    check(to_nfd("été"), "été")


def test_plain_spaces():
    check(plain_spaces("prix : 12 €"), "prix : 12 €")


@pytest.mark.parametrize("name", sorted(PERTURBATIONS))
def test_perturbations_leave_plain_ascii_alone_except_case(name):
    # A sentence with no accent, apostrophe or special space is unchanged
    # by every perturbation except upper case.
    text = "le chat dort"
    expected = "LE CHAT DORT" if name == "upper case" else text
    check(PERTURBATIONS[name](text), expected)
