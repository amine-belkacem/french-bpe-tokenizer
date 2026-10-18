"""Tests for the BPE core. Expected values are derived by hand in the comments."""

import random

import pytest

from frbpe.bpe import BPETokenizer, merge_pair
from frbpe.pretokenize import PreTokenizer

A, B, C, D = 97, 98, 99, 100  # byte values of "a", "b", "c", "d"


def check(got, expected):
    assert got == expected, f"ECHEC: obtenu {got!r}, attendu {expected!r}"


# --------------------------------------------------------------- merge_pair
def test_merge_pair_basic():
    check(merge_pair([A, B, C, A, B], (A, B), 256), [256, C, 256])


def test_merge_pair_overlapping_run():
    # [a a a]: positions 0-1 merge, position 2 has no partner left.
    check(merge_pair([A, A, A], (A, A), 256), [256, A])
    # [a a a a]: two disjoint merges.
    check(merge_pair([A, A, A, A], (A, A), 256), [256, 256])


# ---------------------------------------------------------- hand-made corpus
# Classic example "aaabdaaabac", which the GPT-4 regex keeps as one chunk.
#
# Step 1, pair counts on a a a b d a a a b a c:
#   (a,a)=4 [positions 0-1, 1-2, 5-6, 6-7], (a,b)=2, (b,d)=1, (d,a)=1,
#   (b,a)=1, (a,c)=1   -> merge (a,a) -> 256
#   word becomes [256, a, b, d, 256, a, b, a, c]
# Step 2: (256,a)=2, (a,b)=2, others 1. Tie at 2, broken by the smallest
#   pair: (97,98) < (256,97) -> merge (a,b) -> 257
#   word becomes [256, 257, d, 256, 257, a, c]
# Step 3: (256,257)=2, others 1 -> merge (256,257) -> 258
#   word becomes [258, d, 258, a, c]
# Step 4: every pair is seen once, below min_pair_count=2 -> training stops.
def test_train_hand_example_merges():
    tok = BPETokenizer.train(["aaabdaaabac"], vocab_size=300)
    check(tok.merges, [(A, A), (A, B), (256, 257)])
    check(tok.vocab_size, 259)


def test_train_hand_example_encoding_and_vocab():
    tok = BPETokenizer.train(["aaabdaaabac"], vocab_size=300)
    check(tok.encode("aaabdaaabac"), [258, D, 258, A, C])
    check(tok.vocab[258], b"aaab")


def test_vocab_size_limit_is_respected():
    # Asking for 257 tokens leaves room for exactly one merge, the first one.
    tok = BPETokenizer.train(["aaabdaaabac"], vocab_size=257)
    check(tok.merges, [(A, A)])


def test_chunk_frequencies_are_weighted():
    # The GPT-4 regex splits "ab ab ab cd" into "ab", " ab", " ab", " cd".
    # (a,b) is seen 3 times (once per chunk), (" ",a) twice, (" ",c) and
    # (c,d) once each. With min_pair_count=3 only (a,b) qualifies.
    tok = BPETokenizer.train(["ab ab ab cd"], vocab_size=300, min_pair_count=3)
    check(tok.merges, [(A, B)])


def test_merges_never_cross_chunks():
    # A digit can never be the optional prefix of a word in the GPT-4 regex,
    # so "1a1a1a1a" is cut into "1", "a", "1", "a", ... The pair ("1", "a")
    # never exists inside a chunk, so nothing can be learned.
    # (My first version used "a.a.a.a." and was wrong: a punctuation mark *can*
    # prefix a word, which gives the chunks "a", ".a", ".a", ".a", ".".)
    check(PreTokenizer("gpt4").split("1a1a"), ["1", "a", "1", "a"])
    tok = BPETokenizer.train(["1a1a1a1a"], vocab_size=300)
    check(tok.merges, [])


def test_vocab_size_below_bytes_is_rejected():
    with pytest.raises(ValueError):
        BPETokenizer.train(["abc"], vocab_size=100)


# ----------------------------------------------------------- pre-tokenizer
@pytest.mark.parametrize(
    "text, expected",
    [
        # The English contraction branch does not match "'é", so the
        # apostrophe is glued to the next word. This is what J2 will fix.
        ("L'école d'été", ["L", "'école", " d", "'été"]),
        ("aujourd'hui", ["aujourd", "'hui"]),
        # Digits go by groups of at most 3, the space is its own chunk.
        ("1 000,50", ["1", " ", "000", ",", "50"]),
        ("peut-être", ["peut", "-être"]),
    ],
)
def test_gpt4_pretokenizer_on_french(text, expected):
    check(PreTokenizer("gpt4").split(text), expected)


def test_unknown_pattern_is_rejected():
    with pytest.raises(ValueError):
        PreTokenizer("klingon")


# --------------------------------------------------------------- round trip
@pytest.fixture(scope="module")
def small_tokenizer():
    text = "Le petit chat est sur le tapis. Les petits chats sont sur les tapis. " * 20
    return BPETokenizer.train([text], vocab_size=400)


TRICKY = [
    "",
    "é",
    "é",                       # NFD: "e" + combining accent, never NFC-ed
    "L’été, l'été",                  # typographic and straight apostrophes
    "prix : 12 €",          # no-break and narrow no-break spaces
    "👍🏽 日本語 Ελληνικά",            # 4-byte emoji with skin tone, CJK, Greek
    "\n\n\t  \r\n",
    "a" * 5000,
]


@pytest.mark.parametrize("text", TRICKY)
def test_round_trip_tricky(small_tokenizer, text):
    check(small_tokenizer.decode(small_tokenizer.encode(text)), text)


def test_round_trip_random_unicode(small_tokenizer):
    rng = random.Random(1234)
    for _ in range(200):
        # Any code point outside the surrogate range is valid in UTF-8.
        chars = []
        for _ in range(rng.randint(0, 40)):
            cp = rng.randint(0, 0x10FFFF)
            if 0xD800 <= cp <= 0xDFFF:
                cp = 0x20
            chars.append(chr(cp))
        text = "".join(chars)
        check(small_tokenizer.decode(small_tokenizer.encode(text)), text)


def test_unseen_character_falls_back_to_bytes(small_tokenizer):
    # "龍" is 3 bytes in UTF-8 (E9 BE 8D) and never appears in training,
    # so it can only be encoded as its 3 raw bytes.
    check(small_tokenizer.encode("龍"), [0xE9, 0xBE, 0x8D])


def test_encoding_uses_learned_merges(small_tokenizer):
    # "petit" is frequent in training: it must take fewer tokens than bytes.
    n = len(small_tokenizer.encode(" petit"))
    assert n < len(" petit".encode()), f"ECHEC: obtenu {n} tokens, attendu moins de 6"


# ------------------------------------------------- determinism / persistence
def test_training_is_deterministic():
    texts = ["Les élèves étudient l'été. " * 30, "Le soleil brille sur la mer. " * 30]
    first = BPETokenizer.train(texts, vocab_size=350)
    second = BPETokenizer.train(list(reversed(texts)), vocab_size=350)
    check(second.fingerprint(), first.fingerprint())


def test_save_load_round_trip(tmp_path, small_tokenizer):
    path = tmp_path / "model.bpe"
    small_tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    check(loaded.merges, small_tokenizer.merges)
    text = "Les petits chats sur le tapis."
    check(loaded.encode(text), small_tokenizer.encode(text))


def test_load_rejects_truncated_file(tmp_path, small_tokenizer):
    path = tmp_path / "model.bpe"
    small_tokenizer.save(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        BPETokenizer.load(path)


# ------------------------------------------- incremental vs naive training
def naive_train(texts, vocab_size, min_pair_count=2):
    """Slow reference: recount every pair from scratch after each merge.
    Same tie-break rule as the real trainer (highest count, then smallest pair)."""
    from collections import Counter

    pre = PreTokenizer("gpt4")
    chunks = Counter(c.encode("utf-8") for t in texts for c in pre.split(t))
    words = {chunk: list(chunk) for chunk in chunks}
    merges = []
    for new_id in range(256, vocab_size):
        counts = Counter()
        for chunk, word in words.items():
            for pair in zip(word, word[1:]):
                counts[pair] += chunks[chunk]
        if not counts:
            break
        pair, count = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if count < min_pair_count:
            break
        merges.append(pair)
        words = {chunk: merge_pair(word, pair, new_id) for chunk, word in words.items()}
    return merges


def test_incremental_training_matches_naive_reference():
    rng = random.Random(7)
    syllables = ["ma", "son", "é", "lé", "ve", "qu'", "il", "s'", "est", " ", " ", ". ", "aa"]
    texts = ["".join(rng.choice(syllables) for _ in range(400)) for _ in range(5)]
    fast = BPETokenizer.train(texts, vocab_size=600).merges
    slow = naive_train(texts, vocab_size=600)
    check(len(fast), len(slow))
    check(fast, slow)
