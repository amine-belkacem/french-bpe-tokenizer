"""Byte-level BPE: training, encoding, decoding, save and load.

Vocabulary layout:
    ids 0..255     the 256 possible byte values (no unknown token is ever needed)
    ids 256..      one id per learned merge, in the order they were learned

A merge (a, b) -> c means "wherever token a is directly followed by token b,
replace the pair with token c". The rank of a merge is c - 256: the earlier it
was learned, the more frequent the pair was, and the higher its priority when
encoding.

Training does not walk over the whole corpus at each step. It first counts how
often each pre-tokenized chunk appears ("chunk" -> frequency), then works on
that table of unique chunks. French Wikipedia has a few hundred thousand
distinct chunks for tens of millions of occurrences, so this is where most of
the speed comes from.
"""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from .pretokenize import PreTokenizer

N_BYTES = 256
FILE_MAGIC = "frbpe v1"
DEFAULT_MIN_PAIR_COUNT = 2

Pair = tuple[int, int]


def merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair``, left to right.

    For ids [a, a, a] and pair (a, a) the result is [aa, a]: once the first two
    are merged, the third "a" has no left neighbour left to merge with.
    """
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def count_chunks(texts: Iterable[str], pretokenizer: PreTokenizer) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for text in texts:
        counts.update(chunk.encode("utf-8") for chunk in pretokenizer.split(text))
    return counts


class _PairStats:
    """Weighted pair counts over the unique chunks, kept up to date after each
    merge instead of being recomputed from scratch.

    ``where[p]`` lists the chunks that contained pair p at some point. It is
    allowed to be stale (a chunk may no longer contain p): it is only used to
    know which chunks to look at, and each one is re-checked.
    """

    def __init__(self, words: list[list[int]], freqs: list[int]) -> None:
        self.words = words
        self.freqs = freqs
        self.counts: dict[Pair, int] = defaultdict(int)
        self.where: dict[Pair, set[int]] = defaultdict(set)
        for idx, word in enumerate(words):
            for pair in zip(word, word[1:]):
                self.counts[pair] += freqs[idx]
                self.where[pair].add(idx)
        # Max-heap through negated counts. Ties are broken by the smallest
        # pair (a, b), which makes training deterministic. Entries become
        # outdated when a count changes; they are skipped when popped.
        self.heap = [(-count, pair) for pair, count in self.counts.items()]
        heapq.heapify(self.heap)

    def best(self) -> tuple[Pair, int] | None:
        while self.heap:
            neg_count, pair = self.heap[0]
            if self.counts.get(pair, 0) == -neg_count:
                return pair, -neg_count
            heapq.heappop(self.heap)  # outdated entry
        return None

    def apply(self, pair: Pair, new_id: int) -> None:
        changed: set[Pair] = set()
        for idx in self.where.pop(pair, ()):
            word = self.words[idx]
            if len(word) < 2:
                continue
            new_word = merge_pair(word, pair, new_id)
            if len(new_word) == len(word):
                continue  # stale index entry, the pair is gone from this chunk
            freq = self.freqs[idx]
            for p in zip(word, word[1:]):
                self.counts[p] -= freq
                changed.add(p)
            for p in zip(new_word, new_word[1:]):
                self.counts[p] += freq
                self.where[p].add(idx)
                changed.add(p)
            self.words[idx] = new_word
        for p in changed:
            count = self.counts[p]
            if count > 0:
                heapq.heappush(self.heap, (-count, p))
            else:
                del self.counts[p]


class BPETokenizer:
    def __init__(self, merges: list[Pair], pattern: str) -> None:
        self.pretokenizer = PreTokenizer(pattern)
        self.merges = list(merges)
        self.ranks: dict[Pair, int] = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab: list[bytes] = [bytes([b]) for b in range(N_BYTES)]
        for a, b in self.merges:
            if a >= len(self.vocab) or b >= len(self.vocab):
                raise ValueError(f"merge ({a}, {b}) uses a token that does not exist yet")
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ----------------------------------------------------------------- training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        pattern: str = "gpt4",
        min_pair_count: int = DEFAULT_MIN_PAIR_COUNT,
        verbose: bool = False,
    ) -> BPETokenizer:
        """Learn ``vocab_size - 256`` merges. Stops earlier if no pair appears
        at least ``min_pair_count`` times: merging a pair seen once only
        memorises the training text."""
        if vocab_size < N_BYTES:
            raise ValueError(f"vocab_size must be >= {N_BYTES}, got {vocab_size}")
        pretokenizer = PreTokenizer(pattern)
        chunk_counts = count_chunks(texts, pretokenizer)
        # Sorted so the result does not depend on the order chunks were seen.
        unique = sorted(chunk_counts)
        words = [list(chunk) for chunk in unique]
        freqs = [chunk_counts[chunk] for chunk in unique]
        stats = _PairStats(words, freqs)

        merges: list[Pair] = []
        for new_id in range(N_BYTES, vocab_size):
            found = stats.best()
            if found is None or found[1] < min_pair_count:
                break
            pair, count = found
            stats.apply(pair, new_id)
            merges.append(pair)
            if verbose and len(merges) % 1000 == 0:
                print(f"  {len(merges):6d} merges, last pair seen {count} times")
        return cls(merges, pattern)

    # ----------------------------------------------------------------- encoding
    def _encode_chunk(self, chunk: bytes) -> list[int]:
        ids = list(chunk)
        while len(ids) >= 2:
            # Apply the highest-priority merge (lowest rank) present in the chunk.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            rank = self.ranks.get(pair)
            if rank is None:
                break
            ids = merge_pair(ids, pair, N_BYTES + rank)
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self.pretokenizer.split(text):
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk.encode("utf-8"))
                self._cache[chunk] = cached
            ids.extend(cached)
        return ids

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.vocab[i] for i in ids)

    def decode(self, ids: Iterable[int]) -> str:
        """A token can hold half of a multi-byte character, so only a full
        sequence is guaranteed to decode. A cut sequence shows U+FFFD."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_str(self, token_id: int) -> str:
        """Readable form of one token, for printing examples of segmentation."""
        return self.vocab[token_id].decode("utf-8", errors="backslashreplace")

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Plain text, one merge per line: easy to diff and to read from C."""
        lines = [FILE_MAGIC, f"pattern {self.pretokenizer.name}", f"merges {len(self.merges)}"]
        lines += [f"{a} {b}" for a, b in self.merges]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if len(lines) < 3 or lines[0] != FILE_MAGIC:
            raise ValueError(f"{path}: not a {FILE_MAGIC!r} file")
        pattern = lines[1].removeprefix("pattern ")
        n_merges = int(lines[2].removeprefix("merges "))
        merge_lines = lines[3:]
        if len(merge_lines) != n_merges:
            raise ValueError(f"{path}: header says {n_merges} merges, found {len(merge_lines)}")
        merges = []
        for line in merge_lines:
            a, b = line.split()
            merges.append((int(a), int(b)))
        return cls(merges, pattern)

    def fingerprint(self) -> str:
        """SHA-256 of the merges and pattern, used to check determinism."""
        data = self.pretokenizer.name + "\n" + "\n".join(f"{a} {b}" for a, b in self.merges)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
