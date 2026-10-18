"""Turn the raw downloads into the three corpora used everywhere else.

Outputs, in data/processed/ (one JSON object per line, {"id": ..., "text": ...}):

* train_wiki.jsonl      Wikipedia articles used to train every tokenizer
* test_wiki.jsonl       held-out Wikipedia articles (same domain)
* test_gutenberg.jsonl  19th century novels cut into ~20k character documents
                        (different domain, never seen during training)
* MANIFEST.json         sizes, document counts and SHA-256 of each output

Design decisions worth knowing:

* The train/test split is done per article, from a hash of the article id.
  Splitting per line would put sentences of the same article on both sides and
  make the test set look easier than it is.
* Every text is NFC-normalised. Without it, "é" can be one code point or "e" +
  combining accent, and the tokenizer would learn both spellings. Robustness to
  NFD input is measured separately later, on purpose.
* Gutenberg files are hard-wrapped at ~70 columns. I re-join the lines of each
  paragraph, otherwise the novels would contain a newline every 10 words and
  the comparison with Wikipedia would mostly measure line breaks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path

TEST_PERCENT = 2            # share of Wikipedia articles kept for the test set
MIN_ARTICLE_CHARS = 500     # shorter pages are mostly stubs and lists
GUTENBERG_DOC_CHARS = 20_000
DEFAULT_TRAIN_MB = 100
DEFAULT_TEST_MB = 5

_GUTENBERG_START = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*$", re.M)
_GUTENBERG_END = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*$", re.M)
_BLANK_LINES = re.compile(r"\n\s*\n")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """NFC, Unix newlines, no trailing spaces, at most one empty line in a row."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def is_test_article(article_id: str, test_percent: int = TEST_PERCENT) -> bool:
    """Deterministic split: the same id always lands on the same side, on any
    machine and in any Python version (unlike the built-in hash())."""
    bucket = int(hashlib.sha256(article_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return bucket < test_percent


def strip_gutenberg_boilerplate(text: str) -> str:
    """Keep only what lies between the START and END markers of the licence."""
    start = _GUTENBERG_START.search(text)
    end = _GUTENBERG_END.search(text)
    if start is None or end is None or end.start() <= start.end():
        raise ValueError("Project Gutenberg START/END markers not found")
    return text[start.end():end.start()]


def unwrap_paragraphs(text: str) -> list[str]:
    """Split on blank lines and join the hard-wrapped lines of each paragraph."""
    paragraphs = []
    for block in _BLANK_LINES.split(text):
        joined = " ".join(line.strip() for line in block.split("\n") if line.strip())
        if joined:
            paragraphs.append(joined)
    return paragraphs


def chunk_paragraphs(paragraphs: Iterable[str], max_chars: int) -> Iterator[str]:
    """Group consecutive paragraphs into documents of about ``max_chars``.
    A paragraph is never cut, so a document can exceed the limit when one
    paragraph alone is longer than it."""
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if current and size + len(paragraph) > max_chars:
            yield "\n\n".join(current)
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2  # +2 for the "\n\n" separator
    if current:
        yield "\n\n".join(current)


def write_jsonl(path: Path, docs: Iterable[tuple[str, str]]) -> dict:
    count = 0
    n_bytes = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for doc_id, text in docs:
            f.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
            count += 1
            n_bytes += len(text.encode("utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"documents": count, "text_bytes": n_bytes, "sha256": digest}


def read_wikipedia(parquet_path: Path) -> Iterator[tuple[str, str]]:
    import pyarrow.parquet as pq  # only needed here, keeps the tests light

    table = pq.ParquetFile(parquet_path)
    for batch in table.iter_batches(columns=["id", "text"], batch_size=2048):
        ids = batch.column("id").to_pylist()
        texts = batch.column("text").to_pylist()
        yield from zip(ids, texts)


def split_wikipedia(
    articles: Iterable[tuple[str, str]], train_bytes: int, test_bytes: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Fill the train and test sets in file order until both budgets are met."""
    train, test = [], []
    train_size = test_size = 0
    for article_id, raw in articles:
        if train_size >= train_bytes and test_size >= test_bytes:
            break
        text = clean_text(raw)
        if len(text) < MIN_ARTICLE_CHARS:
            continue
        size = len(text.encode("utf-8"))
        if is_test_article(article_id):
            if test_size < test_bytes:
                test.append((f"wiki:{article_id}", text))
                test_size += size
        elif train_size < train_bytes:
            train.append((f"wiki:{article_id}", text))
            train_size += size
    return train, test


def read_gutenberg(directory: Path) -> Iterator[tuple[str, str]]:
    for path in sorted(directory.glob("*.txt")):
        raw = path.read_text(encoding="utf-8-sig")
        body = clean_text(strip_gutenberg_boilerplate(raw))
        for k, doc in enumerate(chunk_paragraphs(unwrap_paragraphs(body), GUTENBERG_DOC_CHARS)):
            yield f"gutenberg:{path.stem}:{k:04d}", doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build train/test corpora from data/raw/")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-mb", type=float, default=DEFAULT_TRAIN_MB)
    parser.add_argument("--test-mb", type=float, default=DEFAULT_TEST_MB)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    wiki_path = args.raw / "frwiki_20231101_shard11.parquet"
    if not wiki_path.exists():
        print(f"missing {wiki_path}, run scripts/download_data.py first", file=sys.stderr)
        return 1

    train, test = split_wikipedia(
        read_wikipedia(wiki_path),
        train_bytes=int(args.train_mb * 1e6),
        test_bytes=int(args.test_mb * 1e6),
    )
    manifest = {
        "train_wiki": write_jsonl(args.out / "train_wiki.jsonl", train),
        "test_wiki": write_jsonl(args.out / "test_wiki.jsonl", test),
        "test_gutenberg": write_jsonl(
            args.out / "test_gutenberg.jsonl", read_gutenberg(args.raw / "gutenberg")
        ),
        "parameters": {
            "test_percent": TEST_PERCENT,
            "min_article_chars": MIN_ARTICLE_CHARS,
            "gutenberg_doc_chars": GUTENBERG_DOC_CHARS,
            "train_mb": args.train_mb,
            "test_mb": args.test_mb,
        },
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name in ("train_wiki", "test_wiki", "test_gutenberg"):
        info = manifest[name]
        print(f"{name:15s} {info['documents']:7d} docs  {info['text_bytes'] / 1e6:8.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
