"""Milestone 2: does a French-aware regex reduce the number of tokens?

For every model in models/ and both test sets, counts tokens per document,
then writes:

* results/ablation_fertility.csv  fertility and bytes/token with 95 % CIs
* results/ablation_vs_gpt4.csv    paired relative change vs the GPT-4 regex
                                  at the same vocabulary size
* figures/ablation_rules_32k.png  effect of each French rule alone, 32k
* figures/fertility_vs_vocab.png  fertility against vocabulary size

All models are trained on the same 100 MB of Wikipedia; only the regex and
the vocabulary size change.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from frbpe.bpe import BPETokenizer  # noqa: E402
from frbpe.metrics import count_words, paired_relative_change, ratio_of_sums  # noqa: E402

TEST_SETS = {
    "wiki": Path("data/processed/test_wiki.jsonl"),
    "gutenberg": Path("data/processed/test_gutenberg.jsonl"),
}
RULE_LABELS = {
    "fr_elision": "elision only",
    "fr_number": "numbers only",
    "fr_punct": "no-break space\nbefore punctuation",
    "fr": "all French rules",
}
N_BOOT = 1000


def load_texts(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def tokens_per_doc(tok: BPETokenizer, texts: list[str]) -> np.ndarray:
    return np.array([len(tok.encode(t)) for t in texts], dtype=np.int64)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rules(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, test_set in zip(axes, TEST_SETS):
        sel = [r for r in rows if r["vocab"] == 32000 and r["test_set"] == test_set]
        sel.sort(key=lambda r: list(RULE_LABELS).index(r["pattern"]))
        values = np.array([100 * r["change"] for r in sel])
        err = np.array([
            [100 * (r["change"] - r["change_low"]) for r in sel],
            [100 * (r["change_high"] - r["change"]) for r in sel],
        ])
        y = np.arange(len(sel))
        colors = ["#2a6f97" if r["pattern"] == "fr" else "#89b0c9" for r in sel]
        ax.barh(y, values, xerr=err, color=colors, capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y, [RULE_LABELS[r["pattern"]] for r in sel])
        ax.set_title(f"test set: {test_set}")
        ax.set_xlabel("change in token count vs GPT-4 regex (%)")
        ax.invert_yaxis()
    fig.suptitle("Effect of each French pre-tokenization rule (32k vocab, 95 % CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_fertility(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, test_set in zip(axes, TEST_SETS):
        for pattern, color in (("gpt4", "#c44e52"), ("fr", "#2a6f97")):
            sel = sorted(
                (r for r in rows if r["pattern"] == pattern and r["test_set"] == test_set),
                key=lambda r: r["vocab"],
            )
            if len(sel) < 2:
                continue
            x = [r["vocab"] / 1000 for r in sel]
            y = [r["fertility"] for r in sel]
            yerr = [
                [r["fertility"] - r["fertility_low"] for r in sel],
                [r["fertility_high"] - r["fertility"] for r in sel],
            ]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, color=color,
                        label="GPT-4 regex" if pattern == "gpt4" else "French regex")
        ax.set_title(f"test set: {test_set}")
        ax.set_xlabel("vocabulary size (thousands)")
        ax.set_ylabel("tokens per word")
        ax.set_xticks([8, 16, 32])
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Fertility against vocabulary size (95 % CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    args.results.mkdir(exist_ok=True)
    args.figures.mkdir(exist_ok=True)

    model_paths = sorted(args.models.glob("*.bpe"))
    if not model_paths:
        print(f"no model in {args.models}", file=sys.stderr)
        return 1

    texts = {name: load_texts(path) for name, path in TEST_SETS.items()}
    words = {name: np.array([count_words(t) for t in ts]) for name, ts in texts.items()}
    n_bytes = {name: np.array([len(t.encode("utf-8")) for t in ts]) for name, ts in texts.items()}

    counts: dict[tuple[str, int, str], np.ndarray] = {}
    fertility_rows = []
    for path in model_paths:
        tok = BPETokenizer.load(path)
        pattern, vocab = tok.pretokenizer.name, tok.vocab_size
        # Models trained for "32000" have exactly 32000 entries; round for labels.
        vocab = int(round(vocab / 1000) * 1000)
        for test_set, ts in texts.items():
            tokens = tokens_per_doc(tok, ts)
            counts[(pattern, vocab, test_set)] = tokens
            fert = ratio_of_sums(tokens, words[test_set], n_boot=N_BOOT)
            bpt = ratio_of_sums(n_bytes[test_set], tokens, n_boot=N_BOOT)
            fertility_rows.append({
                "pattern": pattern, "vocab": vocab, "test_set": test_set,
                "documents": len(ts), "words": int(words[test_set].sum()),
                "tokens": int(tokens.sum()),
                **fert.as_dict("fertility"), **bpt.as_dict("bytes_per_token"),
            })
            print(f"{path.name:22s} {test_set:10s} fertility {fert.value:.4f} "
                  f"[{fert.low:.4f}, {fert.high:.4f}]  bytes/token {bpt.value:.3f}")

    change_rows = []
    for (pattern, vocab, test_set), tokens in sorted(counts.items()):
        ref = counts.get(("gpt4", vocab, test_set))
        if pattern == "gpt4" or ref is None:
            continue
        change = paired_relative_change(ref, tokens, n_boot=N_BOOT)
        change_rows.append({"pattern": pattern, "vocab": vocab, "test_set": test_set,
                            **change.as_dict("change")})

    write_csv(args.results / "ablation_fertility.csv", fertility_rows)
    write_csv(args.results / "ablation_vs_gpt4.csv", change_rows)
    plot_rules(change_rows, args.figures / "ablation_rules_32k.png")
    plot_fertility(fertility_rows, args.figures / "fertility_vs_vocab.png")
    for r in change_rows:
        print(f"{r['pattern']:11s} {r['vocab']:6d} {r['test_set']:10s} "
              f"{100 * r['change']:+.2f} % [{100 * r['change_low']:+.2f}, {100 * r['change_high']:+.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
