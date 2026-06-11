#!/usr/bin/env python3
"""Phase 1 — Reanalysis of experiment_001: per-seed paired differences vs vanilla.

Computes:
  – per-seed paired ΔPPL for every condition vs vanilla (same model, same seed)
  – mean paired Δ across seeds
  – sign consistency across seeds
  – Wilcoxon signed-rank p-value (with explicit caveat that n=3 is underpowered)

Usage:
  python white_paper/analysis/paired_analysis.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXP_PATH = _REPO_ROOT / "white_paper" / "results" / "experiment_001" / "report_paper.json"


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_results() -> list[dict[str, Any]]:
    data = json.loads(_EXP_PATH.read_text())
    return data["results"]


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
    condition: dict[str, Any], vanilla: dict[str, Any],
) -> float:
    """Paired Δ: condition_delta_ppl - vanilla_delta_ppl."""
    return condition["delta_ppl"] - vanilla["delta_ppl"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    results = load_results()
    groups = group_by_model_seed(results)

    model_labels = sorted({k[0] for k in groups})
    regularizers = sorted({r["regularizer"] for r in results if r["regularizer"] != "vanilla"})

    print("=" * 74)
    print("  PAIRED ANALYSIS: ΔPPL vs vanila (per-seed, paired by seed)")
    print("=" * 74)

    all_tables: list[str] = []

    for model in model_labels:
        print(f"\n  Model: {model}")
        print(f"  {'─' * 68}")

        header = (
            f"  {'Regimen':<22} "
            f"{'seed=42':>10} {'seed=73':>10} {'seed=137':>10} "
            f"{'mean Δ':>10} {'signs':>12} {'W':>6} {'p-val':>8}"
        )
        print(header)
        print(f"  {'─' * 68}")

        for reg in regularizers:
            diffs: list[float] = []
            per_seed: dict[int, float] = {}

            for seed in [42, 73, 137]:
                key = (model, seed)
                if key not in groups:
                    continue
                cond = groups[key].get(reg)
                vanilla = groups[key].get("vanilla")
                if cond is None or vanilla is None:
                    continue
                d = paired_delta(cond, vanilla)
                diffs.append(d)
                per_seed[seed] = d

            if not diffs:
                continue

            mean_d = sum(diffs) / len(diffs)
            signs = "".join("+" if d > 0 else "–" if d < 0 else "0" for d in diffs)
            w, p = _wilcoxon_sign_rank(diffs)

            print(
                f"  {reg:<22} "
                f"{per_seed.get(42, 0):>+10.2f} "
                f"{per_seed.get(73, 0):>+10.2f} "
                f"{per_seed.get(137, 0):>+10.2f} "
                f"{mean_d:>+10.2f} "
                f"{signs:>12} "
                f"{w:>6.1f} "
                f"{p:>8.4f}",
            )

            all_tables.append(f"| {model:<6} | {reg:<20} | {per_seed.get(42, 0):>+7.2f} | {per_seed.get(73, 0):>+7.2f} | {per_seed.get(137, 0):>+7.2f} | {mean_d:>+7.2f} | {signs:<12} | {p:<8.4f} |")

    print("\n")
    print("=" * 74)
    print("  CAVEAT")
    print("=" * 74)
    print("""
  n=3 seeds per condition.  The Wilcoxon p-value above uses the exact
  distribution, but with n=3 the smallest attainable two-sided p-value
  is 0.25 (2/8 sign assignments).  **Sign consistency (+++ vs ––– vs
  mixed) is the meaningful statistic here, not the p-value.**

  At n=5 (Phase 2, step 7), sign consistency across seeds will become
  the headline statistic, with a pre-registered threshold of ≥4/5 same
  sign for a "reliable effect."
""")

    # Summary table for paper
    print("\n")
    print("=" * 74)
    print("  MARKDOWN TABLE (for paper.md)")
    print("=" * 74)
    print()
    print("| Model | Regimen | seed=42 | seed=73 | seed=137 | mean Δ | signs | p-val |")
    print("|---|---|---|---|---|---|---|---|")
    for t in all_tables:
        print(t)

    return all_tables


if __name__ == "__main__":
    main()
