"""Tests for scripts/prepare_corpus.py.

Each expected value is worked out by hand in the comment above the check.
"""

import hashlib

import pytest

from prepare_corpus import (
    MIN_ARTICLE_CHARS,
    chunk_paragraphs,
    clean_text,
    is_test_article,
    split_wikipedia,
    strip_gutenberg_boilerplate,
    unwrap_paragraphs,
)


def check(got, expected):
    assert got == expected, f"ECHEC: obtenu {got!r}, attendu {expected!r}"


def test_clean_text_nfc_newlines_and_spaces():
    # "e" + U+0301 (combining acute) becomes the single code point "é" in NFC.
    # "\r\n" -> "\n", trailing spaces removed, 4 newlines collapsed to 2.
    raw = "Café \r\nligne 2  \n\n\n\nfin"
    check(clean_text(raw), "Café\nligne 2\n\nfin")


def test_clean_text_output_is_nfc_single_code_point():
    # "é" in NFC is 1 code point (U+00E9), in NFD it is 2 (U+0065 U+0301).
    check(len(clean_text("é")), 1)


def test_unwrap_paragraphs_joins_hard_wrapped_lines():
    text = "Il était\nune fois.\n\n  Fin\nde l'histoire.\n"
    check(unwrap_paragraphs(text), ["Il était une fois.", "Fin de l'histoire."])


def test_chunk_paragraphs_respects_budget():
    # max 10 chars. "aaaa" -> size 4+2 = 6. "bbbb": 6+4 = 10, not > 10, kept,
    # size 12. "cc": 12+2 = 14 > 10, so the first document is closed.
    # First document "aaaa\n\nbbbb" has exactly 10 chars.
    check(list(chunk_paragraphs(["aaaa", "bbbb", "cc"], 10)), ["aaaa\n\nbbbb", "cc"])


def test_chunk_paragraphs_never_cuts_a_paragraph():
    long_paragraph = "x" * 15
    check(list(chunk_paragraphs([long_paragraph], 10)), [long_paragraph])


def test_strip_gutenberg_boilerplate_keeps_body_only():
    raw = (
        "The Project Gutenberg eBook of Germinal\nlicence...\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK GERMINAL ***\n"
        "Corps du livre.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK GERMINAL ***\n"
        "more licence\n"
    )
    check(strip_gutenberg_boilerplate(raw).strip(), "Corps du livre.")


def test_strip_gutenberg_boilerplate_rejects_unknown_format():
    with pytest.raises(ValueError):
        strip_gutenberg_boilerplate("no markers at all")


def test_is_test_article_hand_computed_bucket():
    # sha256("1") starts with 6b86b273. 0x6b86b273 = 1803989619,
    # 1803989619 mod 100 = 19, and 19 >= 2, so article "1" goes to train.
    check(hashlib.sha256(b"1").hexdigest()[:8], "6b86b273")
    check(int("6b86b273", 16), 1803989619)
    check(is_test_article("1"), False)
    # With a 20 % test share, bucket 19 < 20, so the same article goes to test.
    check(is_test_article("1", test_percent=20), True)


def test_is_test_article_share_is_close_to_target():
    # 2 % of 20000 ids is 400. A hash is not exactly uniform on a finite
    # sample, so I accept 300..500 (a binomial std is about 20 here).
    n_test = sum(is_test_article(str(i)) for i in range(20_000))
    assert 300 <= n_test <= 500, f"ECHEC: obtenu {n_test}, attendu entre 300 et 500"


def test_split_wikipedia_no_leak_and_short_articles_dropped():
    body = "a" * MIN_ARTICLE_CHARS
    articles = [(str(i), body) for i in range(2000)] + [("short", "trop court")]
    train, test = split_wikipedia(articles, train_bytes=10**9, test_bytes=10**9)
    train_ids = {doc_id for doc_id, _ in train}
    test_ids = {doc_id for doc_id, _ in test}
    check(train_ids & test_ids, set())
    check("wiki:short" in train_ids | test_ids, False)
    check(len(train) + len(test), 2000)


def test_split_wikipedia_stops_at_budget():
    body = "a" * 1000  # 1000 bytes per article in UTF-8
    articles = [(str(i), body) for i in range(5000)]
    train, _ = split_wikipedia(articles, train_bytes=10_000, test_bytes=0)
    # The budget is checked before adding, so it stops at exactly 10 articles.
    check(len(train), 10)
