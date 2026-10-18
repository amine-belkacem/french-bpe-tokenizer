"""Compression metrics and their confidence intervals.

Fertility = tokens / words over a whole corpus (a ratio of sums, not a mean
of per-document ratios, so long documents weigh what they contain).

What counts as a word has to be the same for every tokenizer, otherwise the
comparison means nothing. I use the simplest rule I can defend: a word is a
maximal run of letters, combining marks and digits. "l'école" is 2 words,
"aujourd'hui" is 2 words, "1 000" is 2 words. The rule is debatable, but:

* it does not depend on any tokenizer, and
* when two tokenizers are compared on the same documents, the word count
  cancels out: fertility_b / fertility_a = tokens_b / tokens_a. The relative
  differences reported by the ablation do not depend on this rule at all.

Confidence intervals come from a bootstrap over documents: draw as many
documents as there are, with replacement, recompute the ratio, repeat, and
keep the 2.5 % and 97.5 % percentiles. Resampling documents rather than
tokens keeps the fact that tokens of one document are not independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import regex

WORD_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+")
DEFAULT_N_BOOT = 1000
DEFAULT_LEVEL = 0.95


def count_words(text: str) -> int:
    return sum(1 for _ in WORD_RE.finditer(text))


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float

    def as_dict(self, prefix: str) -> dict[str, float]:
        return {prefix: self.value, f"{prefix}_low": self.low, f"{prefix}_high": self.high}


def _resample_indices(n_docs: int, n_boot: int, seed: int) -> np.ndarray:
    if n_docs == 0:
        raise ValueError("cannot bootstrap an empty corpus")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_docs, size=(n_boot, n_docs))


def _percentiles(samples: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high)


def ratio_of_sums(
    numerator,
    denominator,
    n_boot: int = DEFAULT_N_BOOT,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
) -> Estimate:
    """sum(numerator) / sum(denominator) with a bootstrap interval over documents."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    if num.shape != den.shape:
        raise ValueError(f"shape mismatch: {num.shape} vs {den.shape}")
    idx = _resample_indices(len(num), n_boot, seed)
    samples = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    low, high = _percentiles(samples, level)
    return Estimate(float(num.sum() / den.sum()), low, high)


def paired_relative_change(
    tokens_ref,
    tokens_new,
    n_boot: int = DEFAULT_N_BOOT,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
) -> Estimate:
    """Relative change in token count, new vs reference, on the same documents:
    sum(new) / sum(ref) - 1. Negative means the new tokenizer needs fewer
    tokens. The same resampled documents are used for both tokenizers, which
    is what makes the interval much tighter than comparing two separate CIs."""
    ref = np.asarray(tokens_ref, dtype=np.float64)
    new = np.asarray(tokens_new, dtype=np.float64)
    if ref.shape != new.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {new.shape}")
    idx = _resample_indices(len(ref), n_boot, seed)
    samples = new[idx].sum(axis=1) / ref[idx].sum(axis=1) - 1.0
    low, high = _percentiles(samples, level)
    return Estimate(float(new.sum() / ref.sum() - 1.0), low, high)
