#!/usr/bin/env python3
"""Run standard NLU benchmarks on SmolLM2-1.7B with LoRA variants.

Each variant is a *trained* soul directory (see `rengongtong save`).  The
previous version compared the base model against a freshly initialised
"mpt" Brain, whose LoRA B matrices are zero, so both rows were identical
by construction.

Usage:
  python white_paper/evaluate_benchmarks.py --soul mpt=souls/abc--2026... --soul vanilla=souls/def--2026...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure rengongtong package is accessible
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

from rengongtong.brain import Brain
from rengongtong.synaptic import BASE_MODEL
import torch

from rengongtong.benchmark import HellaSwagEval, ARCEval, MMLUProEval


def build_variants(souls: list[str]) -> list[tuple[str, str, Path | None]]:
    """``["tag=path", ...]`` → [(tag, label, soul_path)], always starting with the base model."""
    variants: list[tuple[str, str, Path | None]] = [("base", "Base model (no LoRA training)", None)]
    for spec in souls:
        tag, sep, path = spec.partition("=")
        if not sep:
            tag, path = Path(spec).name, spec
        soul = Path(path)
        if not any(soul.glob("*.safetensors")):
            raise SystemExit(f"{soul} is not a soul directory (no *.safetensors adapter)")
        variants.append((tag, f"Trained soul {soul}", soul))
    return variants


def run_benchmarks(
    variants: list[tuple[str, str, Path | None]],
    max_samples: int | None = 100,
) -> list[dict]:
    tasks = [HellaSwagEval(), ARCEval(), MMLUProEval(max_categories=3)]

    results = []
    for tag, label, soul_path in variants:
        print(f"\n{'=' * 60}")
        print(f"[{tag}] {label}")
        print(f"{'=' * 60}")

        t0 = time.perf_counter()
        brain = Brain(model_name=BASE_MODEL, soul_path=soul_path)
        model = brain.model
        tokenizer = brain.tokenizer
        t_load = time.perf_counter() - t0
        print(f"  Loaded in {t_load:.1f}s")

        task_results = {}
        for task in tasks:
            t1 = time.perf_counter()
            try:
                task.load()
                metrics = task.evaluate(model, tokenizer, max_samples=max_samples)
                elapsed = time.perf_counter() - t1
                print(f"  {task.name}: {metrics['accuracy']*100:.1f}%  ({metrics['correct']}/{metrics['total']})  [{elapsed:.1f}s]")
                task_results[task.name] = metrics
            except Exception as e:
                print(f"  {task.name}: ERROR — {e}")
                task_results[task.name] = {"error": str(e)}

        results.append({
            "variant": tag,
            "label": label,
            "soul_path": str(soul_path) if soul_path else None,
            "load_time_s": round(t_load, 2),
            "tasks": task_results,
        })

        del brain
        torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--soul", action="append", default=[], help="tag=path/to/soul (repeatable)")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    variants = build_variants(args.soul)
    if len(variants) == 1:
        print("WARNING: no --soul given; only the base model will be evaluated.")

    print("=" * 72)
    print("BENCHMARK EVALUATION — SmolLM2-1.7B")
    print("=" * 72)
    print(f"Note: using max_samples={args.max_samples} per task.")

    results = run_benchmarks(variants, max_samples=args.max_samples)

    out_path = _SCRIPT_DIR / "benchmark_report.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nReport saved to {out_path.resolve()}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in results:
        print(f"\n[{r['variant']}] {r['label']}")
        for name, m in r["tasks"].items():
            if "error" in m:
                print(f"  {name}: ERROR")
            else:
                print(f"  {name}: {m['accuracy']*100:.1f}% ({m['correct']}/{m['total']})")


if __name__ == "__main__":
    main()
