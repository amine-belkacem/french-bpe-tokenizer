"""One interface over frbpe, tiktoken and SentencePiece, for the comparison.

Each adapter exposes the same three things: encode a text into ids, decode
ids back into text, and show each token as readable text for the
segmentation examples. tiktoken and sentencepiece are imported only when an
adapter of that kind is built, so the rest of the package does not need them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .bpe import BPETokenizer

SP_SPACE = "▁"  # "▁", how SentencePiece writes a space inside a piece


def piece_repr(data: bytes) -> str:
    """Readable form of a token's bytes. A byte-level token can hold part of a
    multi-byte character (half of "é"); those bytes are shown as <0xC3>."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"<0x{b:02X}>" for b in data)


@dataclass
class Adapter:
    name: str
    family: str          # "frbpe", "tiktoken" or "sentencepiece"
    vocab_size: int
    trained_on: str      # short description of the training data
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]
    pieces: Callable[[list[int]], list[str]]


def frbpe_adapter(path: str | Path, name: str, trained_on: str) -> Adapter:
    tok = BPETokenizer.load(path)
    return Adapter(
        name=name,
        family="frbpe",
        vocab_size=tok.vocab_size,
        trained_on=trained_on,
        encode=tok.encode,
        decode=tok.decode,
        pieces=lambda ids: [piece_repr(tok.vocab[i]) for i in ids],
    )


def tiktoken_adapter(encoding_name: str, name: str, trained_on: str) -> Adapter:
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    return Adapter(
        name=name,
        family="tiktoken",
        vocab_size=enc.n_vocab,
        trained_on=trained_on,
        # encode_ordinary: plain text, special tokens like <|endoftext|> are
        # not recognised (the corpora never contain them anyway).
        encode=enc.encode_ordinary,
        decode=enc.decode,
        pieces=lambda ids: [piece_repr(enc.decode_single_token_bytes(i)) for i in ids],
    )


def sentencepiece_adapter(path: str | Path, name: str, trained_on: str) -> Adapter:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(path))
    return Adapter(
        name=name,
        family="sentencepiece",
        vocab_size=sp.get_piece_size(),
        trained_on=trained_on,
        encode=lambda text: sp.encode(text, out_type=int),
        decode=sp.decode,
        pieces=lambda ids: [sp.id_to_piece(i).replace(SP_SPACE, " ") for i in ids],
    )
