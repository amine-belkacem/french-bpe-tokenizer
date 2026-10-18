"""The C encoder must give exactly the same chunks and ids as Python.

Skipped when c/build/frbpe_cli has not been built (run `make -C c` first, or
set FRBPE_CLI). The CI builds it, with sanitizers, before running pytest.
"""

import random
from pathlib import Path

import pytest

from c_bridge import c_encode, c_split_ends, cli_path, python_split_ends
from frbpe.bpe import BPETokenizer
from frbpe.pretokenize import PATTERNS, PreTokenizer

pytestmark = pytest.mark.skipif(not cli_path().exists(), reason="C tool not built")

# Pieces chosen to hit every branch of the regex and its corner cases.
PIECES = [
    "a", "é", "e", "l", "s", "t", "q", "u", "i", "h", "L", "Q", "A", "Z", "ç", "Œ",
    "'", "’", " ", " ", " ", " ", " ", "\t", "\n", "\r\n", "\r", "\x0b",
    "0", "1", "42", "1789", "٣",  # Arabic-Indic digit is \p{N} too
    ".", ",", ";", ":", "!", "?", "«", "»", "-", "(", ")", "€", "_", "…",
    "́",            # combining accent: not \p{L}
    "ſ", "İ", "K",  # ſ, İ, Kelvin sign: case-folding corner cases
    "aujourd'hui", "AUJOURD’HUI", "aujourd'hui_", "aujourd'hui5", "aujourd'huis",
    "jusqu'", "lorsqu’", "qu'", "l'", "'s", "'LL", "'re", "'ve", "'t", "'ſ",
    "  ", "\n\n", " \n ", " :", "1 000",
    "日本", "👍🏽", "Ελλάδα",
]


def random_texts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    texts = []
    for _ in range(n):
        parts = [rng.choice(PIECES) for _ in range(rng.randint(0, 25))]
        if rng.random() < 0.2:
            cp = rng.randint(0, 0x10FFFF)
            parts.append(chr(cp) if not 0xD800 <= cp <= 0xDFFF else "x")
        texts.append("".join(parts))
    return texts


TRICKY = [
    "",
    "L'école d'été est fermée aujourd'hui.",
    "Qu’il paie 1 000 € aujourd'hui !",
    "a\n\n  b  ",
    "   ",
    "\n",
    "x" * 3000,
    "été",
]


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
def test_split_matches_python(pattern):
    texts = TRICKY + random_texts(3000, seed=hash(pattern) % 1000)
    pre = PreTokenizer(pattern)
    got = c_split_ends(pattern, texts)
    for text, c_ends in zip(texts, got):
        py_ends = python_split_ends(pre.split(text))
        assert c_ends == py_ends, (
            f"ECHEC pattern={pattern} text={text!r}: obtenu {c_ends}, attendu {py_ends}"
        )


MODELS = Path(__file__).resolve().parent.parent / "models"


# The real 32k models are committed, so the CI can use them too. (A first
# version trained a toy model inside the test; it only learned ~190 merges,
# too few for the comparison to prove much.)
@pytest.fixture(scope="module", params=["gpt4_32k", "fr_32k"])
def trained_model(request):
    path = MODELS / f"{request.param}.bpe"
    return BPETokenizer.load(path), path


def test_encode_matches_python(trained_model):
    tok, path = trained_model
    texts = TRICKY + random_texts(2000, seed=11)
    got = c_encode(path, texts)
    for text, c_ids in zip(texts, got):
        py_ids = tok.encode(text)
        assert c_ids == py_ids, f"ECHEC text={text!r}: obtenu {c_ids}, attendu {py_ids}"


def test_model_has_real_merges(trained_model):
    # Guard against a trivial pass: the comparison must use the full 32k models.
    tok, _ = trained_model
    assert len(tok.merges) == 31744, f"ECHEC: obtenu {len(tok.merges)} merges, attendu 31744"
