"""Download the tokenizers of existing models used as references (milestone 3).

* tiktoken encodings gpt2, cl100k_base, o200k_base (MIT). tiktoken downloads
  them itself and checks each file against a SHA-256 hard-coded in its
  source; I only point its cache to data/reference/tiktoken so nothing lands
  in a hidden system folder.
* SentencePiece models of Mistral 7B v0.1 (Apache 2.0) and CamemBERT base
  (MIT), from Hugging Face at a pinned commit. Their digests are recorded in
  data/reference/SHA256SUMS on the first download and checked afterwards.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from download_data import fetch, load_sums, save_sums

TIKTOKEN_ENCODINGS = ("gpt2", "cl100k_base", "o200k_base")

HF_FILES = {
    "mistral_7b_v01.model": (
        "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/"
        "27d67f1b5f57dc0953326b2601d68371d40ea8da/tokenizer.model"
    ),
    "camembert_base.model": (
        "https://huggingface.co/almanach/camembert-base/resolve/"
        "a75967561c78f2aa81cc41045378d3b4ee25af9e/sentencepiece.bpe.model"
    ),
}

DEFAULT_DIR = Path("data/reference")


def tiktoken_cache_dir(root: Path = DEFAULT_DIR) -> Path:
    return root / "tiktoken"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    sums_path = args.out / "SHA256SUMS"
    sums = load_sums(sums_path)
    for name, url in HF_FILES.items():
        sums[name] = fetch(url, args.out / name, sums.get(name))
    save_sums(sums_path, sums)

    cache = tiktoken_cache_dir(args.out)
    cache.mkdir(exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache.resolve())
    import tiktoken  # imported after setting the cache location

    for name in TIKTOKEN_ENCODINGS:
        enc = tiktoken.get_encoding(name)
        print(f"tiktoken {name}: {enc.n_vocab} tokens, hash checked by tiktoken")
    print(f"reference tokenizers ready in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
