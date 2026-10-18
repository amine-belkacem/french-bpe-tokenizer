"""Compare the C encoder with Python on every document of both test sets.

For each model in models/*.bpe, encodes all test documents with both
implementations and counts the documents whose ids differ. Requirement E6
of the spec is 0 differences. Writes results/c_python_differential.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from c_bridge import c_encode
from frbpe.bpe import BPETokenizer

TEST_SETS = {
    "wiki": Path("data/processed/test_wiki.jsonl"),
    "gutenberg": Path("data/processed/test_gutenberg.jsonl"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results/c_python_differential.csv"))
    args = parser.parse_args(argv)

    texts = {}
    for name, path in TEST_SETS.items():
        with path.open(encoding="utf-8") as f:
            texts[name] = [json.loads(line)["text"] for line in f]

    rows = []
    total_diff = 0
    for model_path in sorted(args.models.glob("*.bpe")):
        tok = BPETokenizer.load(model_path)
        for name, docs in texts.items():
            c_ids = c_encode(model_path, docs)
            diff = 0
            n_tokens = 0
            for doc, ids in zip(docs, c_ids):
                py_ids = tok.encode(doc)
                n_tokens += len(py_ids)
                if ids != py_ids:
                    diff += 1
                    if diff == 1:
                        print(f"first difference in {model_path.name}/{name}: {doc[:80]!r}")
            total_diff += diff
            rows.append({"model": model_path.stem, "test_set": name, "documents": len(docs),
                         "tokens": n_tokens, "documents_different": diff})
            print(f"{model_path.stem:16s} {name:10s} {len(docs):5d} docs {n_tokens:9d} tokens "
                  f"{diff} different")

    args.out.parent.mkdir(exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"total documents with a difference: {total_diff}")
    return 0 if total_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
