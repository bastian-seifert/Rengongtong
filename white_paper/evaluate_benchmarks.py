#!/usr/bin/env python3
"""Run standard NLU benchmarks on SmolLM2-1.7B with LoRA variants."""

from __future__ import annotations

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
from rengongtong._types import MptConfig, DecayMode
import torch

from rengongtong.benchmark import HellaSwagEval, ARCEval, MMLUProEval


VARIANTS = [
    ("base", "Base model (no LoRA training)", None),
    ("mpt", "MPT hybrid + subspace", MptConfig(
        decay_mode=DecayMode.HYBRID,
        subspace_protection_weight=0.05,
        subspace_rank=8,
    )),
]


def run_benchmarks(max_samples: int | None = 100) -> list[dict]:
    tasks = [HellaSwagEval(), ARCEval(), MMLUProEval(max_categories=3)]

    results = []
    for tag, label, mpt_cfg in VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"[{tag}] {label}")
        print(f"{'=' * 60}")

        t0 = time.perf_counter()
        brain = Brain(model_name=BASE_MODEL, mpt=mpt_cfg or MptConfig())
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
            "load_time_s": round(t_load, 2),
            "tasks": task_results,
        })

        del brain
        torch.cuda.empty_cache()

    return results


def main() -> None:
    print("=" * 72)
    print("BENCHMARK EVALUATION — SmolLM2-1.7B")
    print("=" * 72)
    print("Note: using max_samples=100 per task for quick evaluation.")
    print("Set max_samples=None to run full evaluation.")

    import torch
    results = run_benchmarks(max_samples=100)

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
