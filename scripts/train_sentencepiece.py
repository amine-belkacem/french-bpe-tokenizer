"""Train SentencePiece BPE and Unigram on the same corpus as frbpe (milestone 3).

This is the fairest reference I can build: same 100 MB of Wikipedia, same
32k vocabulary, settings chosen to match frbpe as closely as SentencePiece
allows. Only the algorithm changes.

* byte_fallback=True              unseen characters become bytes, like frbpe
* normalization_rule_name=identity  no NFKC: the text is kept as is
* remove_extra_whitespaces=False    spaces are kept, so decoding is lossless
* split_digits=False                numbers are not forced to one token per digit

SentencePiece reads one sentence per line, so each paragraph of the training
documents becomes one line. It never sees a newline during training; at
encoding time a newline goes through byte fallback (one token).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import sentencepiece as spm

MAX_SENTENCE_BYTES = 1 << 16  # default 4192 would silently drop long paragraphs


def write_lines(src: Path, dest: Path) -> int:
    n = 0
    with src.open(encoding="utf-8") as fin, dest.open("w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            for paragraph in json.loads(line)["text"].split("\n"):
                if paragraph.strip():
                    fout.write(paragraph + "\n")
                    n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=Path("data/processed/train_wiki.jsonl"))
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args(argv)

    lines_path = args.train.with_name("sp_train_lines.txt")
    n_lines = write_lines(args.train, lines_path)
    print(f"{n_lines} paragraphs written to {lines_path}")

    for model_type in ("bpe", "unigram"):
        prefix = args.out_dir / f"sp_{model_type}_{args.vocab_size // 1000}k"
        start = time.perf_counter()
        spm.SentencePieceTrainer.train(
            input=str(lines_path),
            model_prefix=str(prefix),
            model_type=model_type,
            vocab_size=args.vocab_size,
            byte_fallback=True,
            normalization_rule_name="identity",
            remove_extra_whitespaces=False,
            split_digits=False,
            max_sentence_length=MAX_SENTENCE_BYTES,
            num_threads=4,
            minloglevel=2,
        )
        elapsed = time.perf_counter() - start
        print(f"sp {model_type}: {elapsed:.1f} s -> {prefix}.model")
        # The .vocab file is a readable copy of what the .model already holds.
        prefix.with_suffix(".vocab").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
