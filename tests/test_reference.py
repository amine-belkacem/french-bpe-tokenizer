"""Tests for the adapters that do not need any downloaded file."""

from frbpe.bpe import BPETokenizer
from frbpe.reference import frbpe_adapter, piece_repr


def check(got, expected):
    assert got == expected, f"ECHEC: obtenu {got!r}, attendu {expected!r}"


def test_piece_repr_valid_utf8():
    check(piece_repr(" école".encode("utf-8")), " école")


def test_piece_repr_half_character():
    # "é" is C3 A9 in UTF-8. A token holding only the first byte cannot be
    # decoded on its own and must be shown as <0xC3>.
    check(piece_repr(b"\xc3"), "<0xC3>")
    check(piece_repr(b"a\xc3"), "<0x61><0xC3>")


def test_frbpe_adapter_pieces_rebuild_the_text(tmp_path):
    tok = BPETokenizer.train(["Le chat dort. Le chat mange. " * 10], vocab_size=300)
    path = tmp_path / "m.bpe"
    tok.save(path)
    adapter = frbpe_adapter(path, name="test", trained_on="toy")
    text = "Le chat dort."
    ids = adapter.encode(text)
    check("".join(adapter.pieces(ids)), text)
    check(adapter.decode(ids), text)
    check(adapter.vocab_size, tok.vocab_size)
