"""Tests for src/frbpe/metrics.py, expected values computed by hand."""

import pytest

from frbpe.metrics import count_words, paired_relative_change, ratio_of_sums


def check(got, expected, tol=1e-12):
    if isinstance(expected, float):
        ok = abs(got - expected) <= tol
    else:
        ok = got == expected
    assert ok, f"ECHEC: obtenu {got!r}, attendu {expected!r}"


def test_count_words_french_rule():
    # Runs of letters/digits: L | école | d | été | 1 | 000  -> 6 words.
    # The comma and the euro sign are not words.
    check(count_words("L'école d'été, 1 000 €"), 6)


def test_count_words_nfd_is_one_word():
    # "e" + U+0301 + "te" + U+0301: combining marks belong to the word.
    check(count_words("été"), 1)


def test_count_words_empty():
    check(count_words(" ,;! "), 0)


def test_ratio_of_sums_point_value():
    # Ratio of sums: (3 + 5) / (1 + 3) = 8 / 4 = 2.0.
    # The mean of per-document ratios would differ: (3/1 + 5/3) / 2 = 2.333...
    est = ratio_of_sums([3, 5], [1, 3], n_boot=200)
    check(est.value, 2.0)


def test_ratio_of_sums_constant_documents_give_a_zero_width_interval():
    # Every document has 2 tokens per word, so every resample gives exactly 2.
    est = ratio_of_sums([4, 6, 10], [2, 3, 5], n_boot=200)
    check((est.value, est.low, est.high), (2.0, 2.0, 2.0))


def test_ratio_of_sums_interval_contains_value():
    est = ratio_of_sums([10, 30, 25, 5, 40], [9, 20, 20, 5, 25], n_boot=500)
    assert est.low <= est.value <= est.high, f"ECHEC: obtenu {est}, attendu low <= value <= high"


def test_bootstrap_is_reproducible_with_a_seed():
    a = ratio_of_sums([10, 30, 25, 5], [9, 20, 20, 5], n_boot=300, seed=42)
    b = ratio_of_sums([10, 30, 25, 5], [9, 20, 20, 5], n_boot=300, seed=42)
    check(a, b)


def test_paired_relative_change_hand_value():
    # Each document needs 10 % fewer tokens: (9 + 18) / (10 + 20) - 1
    # = 27 / 30 - 1 = -0.1. Any resample keeps the same ratio, so the
    # interval collapses on -0.1.
    est = paired_relative_change([10, 20], [9, 18], n_boot=200)
    check(est.value, -0.1)
    check(est.low, -0.1)
    check(est.high, -0.1)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        paired_relative_change([1, 2, 3], [1, 2])


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        ratio_of_sums([], [])
