"""Train a BPE tokenizer on data/processed/train_wiki.jsonl and save it.

Example:
    python scripts/train_tokenizer.py --vocab-size 32000 --pattern gpt4

Writes models/<pattern>_<vocab>.bpe and models/<pattern>_<vocab>.json. The
JSON file keeps the training time and the fingerprint, so a second run can be
compared to the first one.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from frbpe.bpe import BPETokenizer


def iter_texts(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train_wiki.jsonl"))
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--pattern", default="gpt4")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args(argv)

    if not args.train.exists():
        print(f"missing {args.train}, run scripts/prepare_corpus.py first", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    tok = BPETokenizer.train(
        iter_texts(args.train), vocab_size=args.vocab_size, pattern=args.pattern, verbose=True
    )
    elapsed = time.perf_counter() - start

    stem = f"{args.pattern}_{args.vocab_size // 1000}k"
    tok.save(args.out_dir / f"{stem}.bpe")
    info = {
        "pattern": args.pattern,
        "requested_vocab_size": args.vocab_size,
        "vocab_size": tok.vocab_size,
        "merges": len(tok.merges),
        "train_file": str(args.train.as_posix()),
        "train_seconds": round(elapsed, 1),
        "fingerprint": tok.fingerprint(),
        "python": platform.python_version(),
        "machine": platform.processor() or platform.machine(),
    }
    (args.out_dir / f"{stem}.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
