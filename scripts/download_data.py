"""Download the raw corpora used by the project into data/raw/.

Two sources, both with licences that allow redistribution of derived results:

* French Wikipedia, dump of 2023-11-01, as packaged by Wikimedia on Hugging Face
  (dataset ``wikimedia/wikipedia``, licence CC BY-SA 3.0 / GFDL). I take a single
  parquet shard: it is enough text for a 32k vocabulary and keeps training time
  reasonable in pure Python. The commit hash is pinned so the file never changes.
* Eight French novels from Project Gutenberg (public domain in the US and in
  France). They are only used as an out-of-domain test set.

Every file is checked against a SHA-256 digest. The Wikipedia digest comes from
Hugging Face (LFS object id); the Gutenberg digests are recorded on the first
download in data/raw/SHA256SUMS and verified on later runs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_SHARD = "20231101.fr/train-00011-of-00017.parquet"
WIKI_SHA256 = "9f22117719b41a8b4fbf08ba02112fb5395335822904850619a7bbbeedf16512"
WIKI_URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/"
    f"{WIKI_REVISION}/{WIKI_SHARD}"
)

# Gutenberg id -> short name. Titles and language were checked in each file header.
GUTENBERG_BOOKS = {
    17489: "hugo_miserables_t1",
    14155: "flaubert_madame_bovary",
    5711: "zola_germinal",
    17989: "dumas_monte_cristo_t1",
    19657: "hugo_notre_dame_de_paris",
    798: "stendhal_rouge_et_noir",
    4791: "verne_voyage_centre_terre",
    13951: "dumas_trois_mousquetaires",
}
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

CHUNK_SIZE = 1 << 20  # 1 MiB per read while streaming downloads
TIMEOUT_S = 60


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` through a temporary file, so a crash never
    leaves a truncated file that looks complete."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=TIMEOUT_S) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for block in response.iter_content(CHUNK_SIZE):
                f.write(block)
    tmp.replace(dest)


def load_sums(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name] = digest
    return sums


def save_sums(path: Path, sums: dict[str, str]) -> None:
    lines = [f"{digest}  {name}" for name, digest in sorted(sums.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fetch(url: str, dest: Path, expected_sha: str | None) -> str:
    if not dest.exists():
        print(f"downloading {url}")
        download(url, dest)
    else:
        print(f"already present: {dest.name}")
    actual = sha256_of(dest)
    if expected_sha is not None and actual != expected_sha:
        raise RuntimeError(
            f"checksum mismatch for {dest.name}: got {actual}, expected {expected_sha}"
        )
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    sums_path = args.out / "SHA256SUMS"
    sums = load_sums(sums_path)

    wiki_dest = args.out / "frwiki_20231101_shard11.parquet"
    sums[wiki_dest.name] = fetch(WIKI_URL, wiki_dest, WIKI_SHA256)

    gutenberg_dir = args.out / "gutenberg"
    gutenberg_dir.mkdir(exist_ok=True)
    for book_id, name in GUTENBERG_BOOKS.items():
        dest = gutenberg_dir / f"{book_id}_{name}.txt"
        key = f"gutenberg/{dest.name}"
        sums[key] = fetch(GUTENBERG_URL.format(id=book_id), dest, sums.get(key))

    save_sums(sums_path, sums)
    print(f"all files verified, checksums in {sums_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
