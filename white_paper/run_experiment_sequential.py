#!/usr/bin/env python3
"""Experiment 003 — Sequential learning: MPT in the regime it was designed for.

Protocol:
  10 sequential feeds of disjoint 50-example text batches; after each feed,
  measure val PPL on (a) a fixed general corpus and (b) each previously fed
  batch.  Metrics: average forgetting and backward transfer.

Pre-committed hypothesis:
  Davis–Kahan subspace protection, which hurt in the single-task generalisation
  setting (experiment_001), may help here — this is the regime it was designed
  for.  Gershgorin decay should reduce forgetting by keeping individual-task
  updates spectrally compact.

Conditions:
  - vanilla         — standard LoRA, no regularizer
  - mpt_gershgorin  — Gershgorin decay only
  - mpt_subspace    — subspace loss only
  - mpt_full        — Gershgorin + subspace
  - replay          — experience replay (no MPT)
  - replay_mpt      — experience replay + Gershgorin decay
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import torch
from torch.optim import AdamW
from datasets import load_dataset

from rengongtong.brain import Brain
from rengongtong.synaptic import BASE_MODEL
from rengongtong._types import MptConfig, DecayMode as DM, ReplayConfig
from rengongtong._spectral import subspace_rotation_loss
from rengongtong.metabolism import MetabolicLoop

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = _SCRIPT_DIR / "results" / "experiment_003"
ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

N_BATCHES = 10
BATCH_SIZE = 8
N_VAL = 500
N_BATCH_EXAMPLES = 50          # per feed
STEPS_PER_FEED = 5             # gradient steps per batch
FEED_LR = 5e-5
SEEDS = [42, 73, 137]

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
MODEL_LABEL = "1.7B"

# ---------------------------------------------------------------------------
# Regularizer definitions
# ---------------------------------------------------------------------------


@dataclass
class Condition:
    tag: str
    label: str
    mpt_config: MptConfig
    replay_config: ReplayConfig | None = None
    weight_decay: float = 0.0
    lora_dropout: float = 0.0


def _mpt(decay_mode: str, subspace_w: float) -> MptConfig:
    decay_penalty = 0.1 if decay_mode in (
        DM.GERSHGORIN, DM.HYBRID, DM.DIAGONAL_MASS, DM.ROTATED_GERSHGORIN,
    ) else 0.0
    return MptConfig(
        decay_mode=decay_mode,
        gershgorin_penalty=decay_penalty,
        subspace_protection_weight=subspace_w,
        subspace_rank=8,
    )


VANILLA    = Condition("vanilla",        "Vanilla LoRA",
                       MptConfig(decay_mode=DM.NONE))
MPT_GERSH  = Condition("mpt_gershgorin", "Gershgorin decay only",
                       _mpt("gershgorin", 0.0))
MPT_SUB    = Condition("mpt_subspace",   "Subspace protection only",
                       _mpt("none", 0.05))
MPT_FULL   = Condition("mpt_full",       "Gershgorin + subspace",
                       _mpt("hybrid", 0.05))
REPLAY     = Condition("replay",         "Experience replay only",
                       MptConfig(decay_mode=DM.NONE),
                       replay_config=ReplayConfig(capacity=200, replay_ratio=0.3))
REPLAY_MPT = Condition("replay_mpt",     "Replay + Gershgorin decay",
                       _mpt("gershgorin", 0.0),
                       replay_config=ReplayConfig(capacity=200, replay_ratio=0.3))

CONDITIONS = [VANILLA, MPT_GERSH, MPT_SUB, MPT_FULL, REPLAY, REPLAY_MPT]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_wikitext() -> tuple[list[str], list[str]]:
    print("Loading Wikitext-2...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    all_train = [t for t in ds["train"]["text"] if t.strip()]
    all_val = [t for t in ds["validation"]["text"] if t.strip()]
    return all_train[:N_BATCHES * N_BATCH_EXAMPLES], all_val[:N_VAL]


def compute_ppl(model, tokenizer, texts, batch_size=8) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            tok = tokenizer(
                batch_texts, truncation=True, padding=True, max_length=2048,
                return_tensors="pt",
            )
            ids = tok.input_ids.to(model.device)
            mask = tok.attention_mask.to(model.device)
            labels = ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss.item()
            n = (labels != -100).sum().item()
            total_loss += loss * n
            total_tokens += n
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss) if avg_loss < 50 else float("inf")


def _pinned_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--stat"], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
        dirty = ""
    gpu_name = "none"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "dirty": bool(dirty),
        "host": os.uname().nodename,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
    }


# ---------------------------------------------------------------------------
# Sequential training
# ---------------------------------------------------------------------------


def run_condition(
    cond: Condition,
    seed: int,
    batches: list[list[str]],
    val_texts: list[str],
) -> dict[str, Any]:
    tag = f"{cond.tag}/seed{seed}"
    print(f"\n{'=' * 72}")
    print(f"  [{tag}]  {cond.label}")
    print(f"{'=' * 72}")

    torch.manual_seed(seed)
    random.seed(seed)

    brain = Brain(
        model_name=MODEL_NAME,
        mpt=cond.mpt_config,
        lora_dropout=cond.lora_dropout,
    )
    model = brain.model
    tokenizer = brain.tokenizer

    # Setup replay buffer if needed
    if cond.replay_config is not None:
        from rengongtong.memory import ExperienceReplayBuffer
        brain._replay_buffer = ExperienceReplayBuffer(
            capacity=cond.replay_config.capacity,
            strategy=cond.replay_config.strategy,
            mode=cond.replay_config.mode,
            replay_ratio=cond.replay_config.replay_ratio,
        )
    else:
        brain._replay_buffer = None

    # Setup MPT mechanisms
    base_weights = None
    metabolism = None
    mpt_cfg = cond.mpt_config
    if mpt_cfg.subspace_protection_weight > 0:
        base_weights = brain._synapse.get_lora_weights()
    if mpt_cfg.decay_mode in (DM.GERSHGORIN, DM.HYBRID, DM.DIAGONAL_MASS, DM.ROTATED_GERSHGORIN):
        metabolism = MetabolicLoop(
            mode=mpt_cfg.decay_mode,
            gershgorin_penalty=mpt_cfg.gershgorin_penalty,
        )

    params = [p for p in model.parameters() if p.requires_grad]

    # Baseline PPL on general val set
    val_ppl_before = compute_ppl(model, tokenizer, val_texts)
    print(f"  Val PPL before: {val_ppl_before:.2f}")

    # Track per-batch PPL matrix
    # after_feed[i][j] = PPL on batch j after feed i
    batch_texts_list = [" ".join(b) for b in batches]
    n = len(batches)
    after_feed: list[list[float | None]] = [[None] * n for _ in range(n)]
    val_ppl_trace: list[float] = []
    train_ppl_trace: list[float] = []

    for feed_idx in range(n):
        feed_batch = batches[feed_idx]
        feed_text = " ".join(feed_batch)

        t0 = time.perf_counter()

        # Training
        model.train()
        feed_loss = 0.0
        n_steps = 0
        for _ in range(STEPS_PER_FEED):
            batch, _ = brain._synapse.prepare_texts(feed_batch)
            optim = AdamW(params, lr=FEED_LR)
            optim.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            # Subspace protection
            if base_weights is not None:
                current = brain._synapse.get_lora_weights()
                sub_loss = subspace_rotation_loss(base_weights, current, mpt_cfg.subspace_rank)
                loss = loss + mpt_cfg.subspace_protection_weight * sub_loss.to(loss.device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            feed_loss += loss.item()
            n_steps += 1

        # End-of-feed metabolic decay (simulate consolidation)
        if metabolism is not None:
            lora_w = brain._synapse.get_lora_weights()
            decayed = metabolism.tick(lora_w, delta_hours=0.5)
            brain._synapse.set_lora_weights(decayed)

        t_train = time.perf_counter() - t0

        # Eval on general val set
        val_ppl = compute_ppl(model, tokenizer, val_texts)
        val_ppl_trace.append(round(val_ppl, 4))

        # Eval on all batches seen so far
        for j in range(feed_idx + 1):
            # Use the batch's individual texts for evaluation
            ppl = compute_ppl(model, tokenizer, batches[j])
            after_feed[feed_idx][j] = round(ppl, 4)

        # Train PPL on current batch
        train_ppl = compute_ppl(model, tokenizer, feed_batch)
        train_ppl_trace.append(round(train_ppl, 4))

        avg_loss = feed_loss / max(n_steps, 1)
        print(
            f"  Feed {feed_idx + 1:>2}/{n}: "
            f"loss={avg_loss:.4f}  "
            f"train_ppl={train_ppl:.2f}  "
            f"val_ppl={val_ppl:.2f}  "
            f"({t_train:.1f}s)"
        )

        # Push to replay buffer
        if brain._replay_buffer is not None:
            brain._replay_buffer.push(feed_text, loss=avg_loss)
            # Sample replay experiences
            replay_loss, n_replay = brain._replay_buffer.replay_loss(
                model, tokenizer, model.device,
            )
            if n_replay > 0:
                optim.zero_grad()
                replay_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()

    # Compute aggregate metrics
    # Backward transfer: for each feed i, PPL on that batch after last feed minus after its own feed
    backward_transfer: list[float | None] = []
    for i in range(n):
        own = after_feed[i][i]
        final = after_feed[n - 1][i]
        if own is not None and final is not None:
            backward_transfer.append(round(final - own, 4))
        else:
            backward_transfer.append(None)

    # Average forgetting: mean of (PPL increase) for all subsequent-eval pairs
    forgetting_deltas: list[float] = []
    for i in range(n):
        for j in range(i):
            if after_feed[i][j] is not None and after_feed[j][j] is not None:
                forgetting_deltas.append(after_feed[i][j] - after_feed[j][j])  # type: ignore[operator]
    avg_forgetting = round(
        sum(forgetting_deltas) / max(len(forgetting_deltas), 1), 4,
    ) if forgetting_deltas else 0.0

    val_ppl_after = compute_ppl(model, tokenizer, val_texts)

    del brain
    torch.cuda.empty_cache()

    result: dict[str, Any] = {
        "tag": tag,
        "condition": cond.tag,
        "seed": seed,
        "model": MODEL_LABEL,
        "val_ppl_before": round(val_ppl_before, 4),
        "val_ppl_after": round(val_ppl_after, 4),
        "delta_ppl": round(val_ppl_after - val_ppl_before, 4),
        "val_ppl_trace": val_ppl_trace,
        "train_ppl_trace": train_ppl_trace,
        "after_feed_matrix": after_feed,
        "backward_transfer": backward_transfer,
        "avg_forgetting": avg_forgetting,
        "mpt_config": asdict(cond.mpt_config),
        "replay_config": asdict(cond.replay_config) if cond.replay_config else None,
        "regularizer_active_components": _active_components(cond),
        "optimizer_config": {"lr": FEED_LR, "betas": (0.9, 0.999), "eps": 1e-8},
    }
    result.update(_pinned_metadata())
    return result


def _active_components(cond: Condition) -> list[str]:
    components: list[str] = []
    mpt_cfg = cond.mpt_config
    if mpt_cfg.decay_mode in (DM.GERSHGORIN, DM.HYBRID, DM.ROTATED_GERSHGORIN):
        components.append("gershgorin_decay")
    if mpt_cfg.decay_mode == DM.DIAGONAL_MASS:
        components.append("diagonal_mass_decay")
    if mpt_cfg.subspace_protection_weight > 0:
        components.append("subspace_protection")
    if cond.replay_config is not None:
        components.append("experience_replay")
    return components


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("  EXPERIMENT 003 — Sequential Learning with MPT")
    print(f"  Root: {ROOT.resolve()}")
    print("=" * 72)

    # Save pinned metadata
    info = _pinned_metadata()
    (ROOT / "commit_info.json").write_text(json.dumps(info, indent=2))

    # Load data
    train_texts_all, val_texts = load_wikitext()
    print(f"  Total train texts: {len(train_texts_all)}  |  Val: {len(val_texts)}")

    # Split into disjoint batches
    random.shuffle(train_texts_all)
    batches: list[list[str]] = [
        train_texts_all[i * N_BATCH_EXAMPLES: (i + 1) * N_BATCH_EXAMPLES]
        for i in range(N_BATCHES)
    ]
    print(f"  Batches: {N_BATCHES} × {N_BATCH_EXAMPLES} texts")

    all_results: list[dict[str, Any]] = []
    total_runs = len(CONDITIONS) * len(SEEDS)
    run_idx = 0

    for cond in CONDITIONS:
        for seed in SEEDS:
            run_idx += 1
            print(f"\n>>> [{run_idx}/{total_runs}]")
            try:
                result = run_condition(cond, seed, batches, val_texts)
                all_results.append(result)
                # Save partial
                (ROOT / "results_partial.json").write_text(
                    json.dumps(all_results, indent=2),
                )
            except Exception as e:
                print(f"\n  ERROR [{cond.tag}/seed{seed}]: {e}")
                import traceback
                traceback.print_exc()
                error_result = {
                    "tag": f"{cond.tag}/seed{seed}",
                    "condition": cond.tag,
                    "seed": seed,
                    "error": str(e),
                }
                all_results.append(error_result)

    # Save final report
    report = {
        "experiment": "mpt-sequential-learning",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": MODEL_LABEL,
            "n_batches": N_BATCHES,
            "batch_size": BATCH_SIZE,
            "n_batch_examples": N_BATCH_EXAMPLES,
            "steps_per_feed": STEPS_PER_FEED,
            "lr": FEED_LR,
            "seeds": SEEDS,
        },
        "conditions": [c.tag for c in CONDITIONS],
        "num_conditions": len(CONDITIONS),
        "num_completed": len(all_results),
        "results": all_results,
    }
    (ROOT / "report_sequential.json").write_text(json.dumps(report, indent=2))
    print(f"\n{'=' * 72}")
    print(f"  Complete. Report: {(ROOT / 'report_sequential.json').resolve()}")
    print(f"  Completed: {len(all_results)}/{total_runs} runs")

    # Summary
    print_summary(all_results)


def print_summary(results: list[dict[str, Any]]):
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if "error" in r:
            continue
        grouped.setdefault(r["condition"], []).append(r)

    print("\n" + "=" * 72)
    print("  SUMMARY — Sequential Learning Metrics")
    print("=" * 72)
    header = (
        f"{'Condition':<20} {'Seed':>5} {'Val ΔPPL':>10} "
        f"{'AvgForget':>10} {'BackTransf':>12}"
    )
    print(header)
    print("-" * len(header))
    for tag in sorted(grouped.keys()):
        for r in grouped[tag]:
            bt = r.get("backward_transfer", [])
            bt_mean = sum(x for x in bt if x is not None) / max(sum(1 for x in bt if x is not None), 1)
            print(
                f"{tag:<20} "
                f"{r['seed']:>5} "
                f"{r['delta_ppl']:>+10.2f} "
                f"{r['avg_forgetting']:>10.4f} "
                f"{bt_mean:>+12.4f}",
            )
        # Mean across seeds
        seeds = grouped[tag]
        valid = [r for r in seeds if "error" not in r]
        if valid:
            mean_d = sum(r["delta_ppl"] for r in valid) / len(valid)
            mean_f = sum(r["avg_forgetting"] for r in valid) / len(valid)
            bt_vals = []
            for r in valid:
                bt = r.get("backward_transfer", [])
                bt_vals.append(sum(x for x in bt if x is not None) / max(sum(1 for x in bt if x is not None), 1))
            mean_bt = sum(bt_vals) / max(len(bt_vals), 1)
            print(
                f"{'  mean':<20} "
                f"{'':5} "
                f"{mean_d:>+10.2f} "
                f"{mean_f:>10.4f} "
                f"{mean_bt:>+12.4f}",
            )
        print("-" * len(header))

    print("=" * 72)


if __name__ == "__main__":
    main()
