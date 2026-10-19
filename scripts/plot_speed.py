"""Plot the benchmark results of the PC and the Raspberry Pi 5 side by side.

Reads results/speed_pc.csv and results/speed_pi5.csv (written by
scripts/benchmark.py) and writes figures/speed.png and figures/memory.png.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MACHINES = {"pc": "PC (Ryzen 3 7320U)", "pi5": "Raspberry Pi 5"}
COLORS = {"pc": "#1baf7a", "pi5": "#6250d6"}  # teal and violet: blue/orange only as a last resort


def load(tag: str) -> dict[str, dict]:
    with (ROOT / "results" / f"speed_{tag}.csv").open(encoding="utf-8") as f:
        return {r["tokenizer"]: r for r in csv.DictReader(f)}


def bars(ax, names, data, key, err_key=None):
    y = np.arange(len(names))
    height = 0.38
    for k, tag in enumerate(MACHINES):
        values = [float(data[tag][n][key]) for n in names]
        errors = [float(data[tag][n][err_key]) for n in names] if err_key else None
        ax.barh(y + (k - 0.5) * height, values, height, xerr=errors, capsize=2,
                color=COLORS[tag], label=MACHINES[tag])
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.legend()


def main() -> int:
    data = {tag: load(tag) for tag in MACHINES}
    names = sorted(data["pc"], key=lambda n: -float(data["pc"][n]["mb_per_s"]))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars(ax, names, data, "mb_per_s", "mb_per_s_std")
    ax.set_xlabel("encoding speed, MB of French text per second (single thread, mean ± std, 10 runs)")
    ax.set_title("Encoding 5 MB of Wikipedia, one document at a time")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "speed.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars(ax, names, data, "peak_rss_mb")
    ax.set_xlabel("peak resident memory of the process (MB), includes the 5 MB corpus")
    ax.set_title("Memory: one process per tokenizer")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "memory.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
