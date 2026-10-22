#!/usr/bin/env python3
"""Paired analysis of an experiment: per-seed differences vs vanilla.

Computes:
  – per-seed paired ΔPPL for every condition vs vanilla (same model, same seed)
  – mean paired Δ across seeds with a 95 % bootstrap confidence interval
  – sign consistency across seeds
  – exact sign-flip permutation test on the mean paired Δ
  – Wilcoxon signed-rank p-value

With n seeds the smallest attainable two-sided p-value of both exact tests
is 2 / 2ⁿ (0.25 for n=3, 0.0625 for n=5, ~0.002 for n=10).

Usage:
  python white_paper/analysis/paired_analysis.py                            # experiment_002
  python white_paper/analysis/paired_analysis.py --experiment experiment_001
  python white_paper/analysis/paired_analysis.py --metric delta_ppl_early_stopped
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = _REPO_ROOT / "white_paper" / "results"


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank (exact distribution for n ≤ 20)
# ---------------------------------------------------------------------------


def _wilcoxon_sign_rank(
    differences: list[float],
) -> tuple[float, float]:
    """Wilcoxon signed-rank test, exact for n ≤ 20.

    Returns (W statistic, two-sided p-value).
    """
    n = len(differences)
    if n == 0:
        return 0.0, 1.0

    # Absolute differences, dropping zeros
    nonzero = [(abs(d), d) for d in differences if abs(d) > 1e-12]
    n_nonzero = len(nonzero)
    if n_nonzero == 0:
        return 0.0, 1.0

    # Rank
    nonzero.sort(key=lambda x: x[0])
    ranks = [
        (rank + 1, d)
        for rank, (_, d) in enumerate(nonzero)
    ]

    W_plus = sum(rank for rank, d in ranks if d > 0)
    W_minus = sum(rank for rank, d in ranks if d < 0)
    W = min(W_plus, W_minus)

    # Exact p-value via enumeration of all 2^n sign assignments
    # (n ≤ 20, so this is fine)
    from itertools import combinations

    total = 2 ** n_nonzero
    count_extreme = 0
    all_ranks = [r for r, _ in ranks]
    total_rank_sum = sum(all_ranks)

    # Enumerate all possible W values for the smaller sign group
    # (We count how many subsets have sum <= W or >= total_rank_sum - W)
    for r in range(n_nonzero + 1):
        for combo in combinations(all_ranks, r):
            w = sum(combo)
            if w <= W or w >= total_rank_sum - W:
                count_extreme += 1

    p_value = count_extreme / total
    return float(W), float(p_value)


def sign_flip_test(differences: list[float]) -> float:
    """Exact two-sided sign-flip permutation test of mean(differences) == 0.

    Enumerates all 2ⁿ sign assignments (n ≤ 20).  Unlike Wilcoxon it uses
    the magnitudes, not just their ranks.
    """
    n = len(differences)
    if n == 0:
        return 1.0
    observed = abs(sum(differences))
    extreme = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        if abs(sum(s * d for s, d in zip(signs, differences))) >= observed - 1e-12:
            extreme += 1
    return extreme / 2 ** n


def bootstrap_ci(
    differences: list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean paired difference."""
    if not differences:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(
        sum(rng.choice(differences) for _ in range(n)) / n
        for _ in range(n_resamples)
    )
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_results(experiment: str) -> list[dict[str, Any]]:
    path = _RESULTS_DIR / experiment / "report_paper.json"
    if not path.exists():
        path = _RESULTS_DIR / experiment / "results_partial.json"
        return [r for r in json.loads(path.read_text()) if "error" not in r]
    return [r for r in json.loads(path.read_text())["results"] if "error" not in r]


def group_by_model_seed(
    results: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    """Group results by (model_label, seed) → {regularizer: result}."""
    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for r in results:
        key = (r["model_label"], r["seed"])
        groups.setdefault(key, {})[r["regularizer"]] = r
    return groups


def paired_delta(
    condition: dict[str, Any], vanilla: dict[str, Any], metric: str = "delta_ppl",
) -> float:
    """Paired Δ: condition[metric] - vanilla[metric]."""
    return condition[metric] - vanilla[metric]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="experiment_002")
    parser.add_argument("--metric", default="delta_ppl", help="delta_ppl | delta_ppl_early_stopped")
    args = parser.parse_args()

    results = load_results(args.experiment)
    groups = group_by_model_seed(results)

    model_labels = sorted({k[0] for k in groups})
    seeds = sorted({k[1] for k in groups})
    regularizers = sorted({r["regularizer"] for r in results if r["regularizer"] != "vanilla"})

    print("=" * 96)
    print(f"  PAIRED ANALYSIS [{args.experiment}]: {args.metric} vs vanilla (paired by seed)")
    print(f"  Seeds: {seeds}   (min attainable two-sided p = {2 / 2 ** len(seeds):.4f})")
    print("=" * 96)

    table_rows: list[str] = []

    for model in model_labels:
        print(f"\n  Model: {model}")
        header = (
            f"  {'Regimen':<22} {'n':>3} {'mean Δ':>9} {'95% CI':>18} "
            f"{'signs':<12} {'p(flip)':>8} {'p(Wilc)':>8}"
        )
        print(header)
        print(f"  {'─' * (len(header) - 2)}")

        for reg in regularizers:
            diffs: list[float] = []
            for seed in seeds:
                group = groups.get((model, seed), {})
                cond, vanilla = group.get(reg), group.get("vanilla")
                if cond is None or vanilla is None or args.metric not in cond:
                    continue
                diffs.append(paired_delta(cond, vanilla, args.metric))
            if not diffs:
                continue

            mean_d = sum(diffs) / len(diffs)
            lo, hi = bootstrap_ci(diffs)
            signs = "".join("+" if d > 0 else "–" if d < 0 else "0" for d in diffs)
            p_flip = sign_flip_test(diffs)
            _, p_wilc = _wilcoxon_sign_rank(diffs)
            ci = f"[{lo:+.2f}, {hi:+.2f}]"
            print(
                f"  {reg:<22} {len(diffs):>3} {mean_d:>+9.2f} {ci:>18} "
                f"{signs:<12} {p_flip:>8.4f} {p_wilc:>8.4f}"
            )
            table_rows.append(
                f"| {model} | {reg} | {len(diffs)} | {mean_d:+.2f} | {ci} | {signs} | {p_flip:.4f} |"
            )

    print("""
  Negative Δ = better than vanilla.  A bootstrap CI over a handful of
  seeds is optimistic; treat it as descriptive.  Pre-registered criterion:
  an effect is called reliable only if p(flip) < 0.05 *and* it beats both
  shrink-matched baselines (weight_decay_matched, uniform_shrink).
""")

    print("| Model | Regimen | n | mean Δ | 95% CI | signs | p(flip) |")
    print("|---|---|---|---|---|---|---|")
    for row in table_rows:
        print(row)

    return table_rows


if __name__ == "__main__":
    main()
