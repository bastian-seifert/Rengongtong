#!/usr/bin/env python3
"""Systematic experiment: MPT spectral regularization for small LMs.

Design (experiment_002, default 5 seeds):
  A — Scale × Regularizer (135M, 360M, 1.7B)
        × (vanilla, weight_decay_matched, uniform_shrink, mpt_gershgorin, mpt_diagonal_mass)
  B — Extra baselines and ablations (1.7B only)
        × (weight_decay, dropout, mpt_full, mpt_subspace)

Changes vs. experiment_001 (see paper.md, "Erratum"):
  * The subspace penalty now uses the live LoRA parameters.  In 001 it was
    computed on detached CPU copies, so it contributed no gradient and
    mpt_subspace / mpt_full trained exactly like vanilla / hybrid.
  * Two shrinkage baselines matched to MPT's per-epoch shrinkage
    (``MATCHED_SHRINK_PER_EPOCH``): AdamW weight decay at a strength that
    actually shrinks (the 0.01 setting shrinks by ~1e-4 over the whole run)
    and plain uniform shrinkage via `DecayMode.UNIFORM`.
  * Train / dev / test split: per-epoch dev PPL selects the early-stopping
    epoch, the held-out Wikitext-2 *test* split is only used for reporting.
  * Per-tick ‖BA‖_F before/after decay is logged so that shrinkage can be
    compared across decay modes.

Usage:
  python white_paper/run_experiment_paper.py                       # full grid
  python white_paper/run_experiment_paper.py --models 135M --seeds 42 73
  python white_paper/run_experiment_paper.py --regularizers vanilla uniform_shrink
"""

from __future__ import annotations

import argparse
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

# Ensure rengongtong package is accessible (repo root is parent of white_paper/)
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
from rengongtong._types import MptConfig, DecayMode as DM
from rengongtong._spectral import (
    _match_lora_pairs,
    gershgorin_lora_instability,
    merged_update_norm,
    principal_right_subspaces,
    spectral_projection_step,
    subspace_leakage_loss,
)
from rengongtong.metabolism import MetabolicLoop

# ---------------------------------------------------------------------------
# Experiment root (relative to this script's location)
# ---------------------------------------------------------------------------

ROOT = _SCRIPT_DIR / "results" / "experiment_002"  # overridden by --experiment

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

N_TRAIN = 100
N_DEV = 250      # Wikitext-2 validation split — early-stopping selection only
N_TEST = 500     # Wikitext-2 test split — reported numbers only
BATCH_SIZE = 8
N_EPOCHS = 15
FEED_LR = 5e-5
SEEDS = [42, 73, 137, 2024, 31337]
DECAY_DELTA_HOURS = 1.0   # MetabolicLoop tick per epoch
DECAY_GAMMA = 0.9         # MetabolicLoop default saliency_gamma

# Fair shrinkage baselines.  MPT's Gershgorin decay scales A and B by
# (1 − λ(t) − γ·instability) once per epoch; with λ₀ = 0.01 and the observed
# instability (0.01–0.3, see experiment_001) that is roughly 0.5–3 % per
# factor and epoch.  Both baselines below target 1 % per factor and epoch;
# the logged ``epoch_shrink`` of the MPT runs shows whether that matches
# (it measures ‖BA‖_F, so 1 % per factor ≈ 2 % there).
MATCHED_SHRINK_PER_EPOCH = 0.01
STEPS_PER_EPOCH = math.ceil(N_TRAIN / BATCH_SIZE)
# Decoupled AdamW decay multiplies by (1 − lr·wd) per step.
MATCHED_WEIGHT_DECAY = MATCHED_SHRINK_PER_EPOCH / (FEED_LR * STEPS_PER_EPOCH)
# MetabolicLoop applies λ(t) = λ₀·exp(−γ·Δt) per tick.
MATCHED_UNIFORM_LAMBDA0 = MATCHED_SHRINK_PER_EPOCH / math.exp(-DECAY_GAMMA * DECAY_DELTA_HOURS)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

@dataclass
class ModelDef:
    name: str
    label: str
    hf_path: str


MODELS = [
    ModelDef(name="SmolLM2-135M",  label="135M",  hf_path="HuggingFaceTB/SmolLM2-135M"),
    ModelDef(name="SmolLM2-360M",  label="360M",  hf_path="HuggingFaceTB/SmolLM2-360M"),
    ModelDef(name="SmolLM2-1.7B",  label="1.7B",  hf_path="HuggingFaceTB/SmolLM2-1.7B"),
]

# ---------------------------------------------------------------------------
# Regularizer definitions
# ---------------------------------------------------------------------------

@dataclass
class RegularizerDef:
    tag: str
    label: str
    mpt_config: MptConfig
    lora_dropout: float = 0.0
    weight_decay: float = 0.0


def _mpt(decay_mode: str, subspace_w: float, base_lambda: float = 0.01) -> MptConfig:
    decay_penalty = 0.1 if decay_mode in (
        DM.GERSHGORIN, DM.HYBRID, DM.DIAGONAL_MASS, DM.ROTATED_GERSHGORIN,
    ) else 0.0
    return MptConfig(
        decay_mode=decay_mode,
        decay_base_lambda=base_lambda,
        gershgorin_penalty=decay_penalty,
        pseudospectral_epsilon=1e-5,
        subspace_protection_weight=subspace_w,
        subspace_rank=8,
    )


VANILLA   = RegularizerDef("vanilla",         "Vanilla LoRA",              MptConfig(decay_mode=DM.NONE, subspace_protection_weight=0.0))
WEIGHT_DECAY = RegularizerDef("weight_decay", "AdamW weight_decay=0.01",   MptConfig(decay_mode=DM.NONE, subspace_protection_weight=0.0), weight_decay=0.01)
WEIGHT_DECAY_MATCHED = RegularizerDef(
    "weight_decay_matched", f"AdamW weight_decay={MATCHED_WEIGHT_DECAY:.2f} (shrink-matched)",
    MptConfig(decay_mode=DM.NONE, subspace_protection_weight=0.0), weight_decay=MATCHED_WEIGHT_DECAY,
)
UNIFORM_SHRINK = RegularizerDef("uniform_shrink", "Uniform shrinkage (shrink-matched)", _mpt("uniform", 0.0, MATCHED_UNIFORM_LAMBDA0))
DROPOUT   = RegularizerDef("dropout",         "LoRA dropout=0.1",          MptConfig(decay_mode=DM.NONE, subspace_protection_weight=0.0), lora_dropout=0.1)
MPT_FULL   = RegularizerDef("mpt_full",       "MPT (hybrid+subspace)",     _mpt("hybrid", 0.05))
MPT_GERSH  = RegularizerDef("mpt_gershgorin", "MPT Gershgorin only",       _mpt("gershgorin", 0.0))
MPT_SUB    = RegularizerDef("mpt_subspace",   "MPT subspace only",         _mpt("none", 0.05))
MPT_DIAGONAL_MASS = RegularizerDef("mpt_diagonal_mass", "MPT diagonal-mass (uniform)", _mpt("diagonal_mass", 0.0))
MPT_ROTATED = RegularizerDef("mpt_rotated",   "MPT rotated-basis Gershgorin", _mpt("rotated_gershgorin", 0.0))

# ---------------------------------------------------------------------------
# Condition matrix
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    model: ModelDef
    reg: RegularizerDef
    seed: int
    tag: str = ""

    def __post_init__(self):
        self.tag = f"{self.model.name}/{self.reg.tag}/seed{self.seed}"


SCALE_REGULARIZERS = [VANILLA, WEIGHT_DECAY_MATCHED, UNIFORM_SHRINK, MPT_GERSH, MPT_DIAGONAL_MASS]
LARGE_ONLY_REGULARIZERS = [WEIGHT_DECAY, DROPOUT, MPT_FULL, MPT_SUB]


def build_conditions(
    seeds: list[int],
    models: set[str] | None = None,
    regularizers: set[str] | None = None,
) -> list[Condition]:
    conds = []
    # A: Scale × Regularizer
    for m in MODELS:
        for r in SCALE_REGULARIZERS:
            for s in seeds:
                conds.append(Condition(m, r, s))
    # B: Extra baselines and component ablation (1.7B only)
    m17 = MODELS[2]
    for r in LARGE_ONLY_REGULARIZERS:
        for s in seeds:
            conds.append(Condition(m17, r, s))
    return [
        c for c in conds
        if (models is None or c.model.label in models or c.model.name in models)
        and (regularizers is None or c.reg.tag in regularizers)
    ]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_wikitext() -> tuple[list[str], list[str], list[str]]:
    """Return (train, dev, test).  Dev selects the early-stopping epoch; test is report-only."""
    print("Loading Wikitext-2...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    all_train = [t for t in ds["train"]["text"] if t.strip()]
    all_dev = [t for t in ds["validation"]["text"] if t.strip()]
    all_test = [t for t in ds["test"]["text"] if t.strip()]
    return all_train[:N_TRAIN], all_dev[:N_DEV], all_test[:N_TEST]


def compute_ppl(model, tokenizer, texts, batch_size=8) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tok = tokenizer(
                batch_texts, truncation=True, padding=True, max_length=2048, return_tensors="pt",
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_condition(
    cond: Condition,
    train_texts: list[str],
    dev_texts: list[str],
    test_texts: list[str],
) -> dict[str, Any]:
    print(f"\n{'─' * 72}")
    print(f"  [{cond.tag}]")
    print(f"  Model: {cond.model.name}  |  Reg: {cond.reg.label}  |  Seed: {cond.seed}")
    print(f"{'─' * 72}")

    # Seed
    torch.manual_seed(cond.seed)
    random.seed(cond.seed)

    t_load_start = time.perf_counter()
    brain = Brain(
        model_name=cond.model.hf_path,
        mpt=cond.reg.mpt_config,
        lora_dropout=cond.reg.lora_dropout,
    )
    model = brain.model
    tokenizer = brain.tokenizer
    t_load = time.perf_counter() - t_load_start

    # Baseline eval
    test_ppl_before = compute_ppl(model, tokenizer, test_texts)
    dev_ppl_before = compute_ppl(model, tokenizer, dev_texts)
    print(f"  Load: {t_load:.1f}s  |  Dev PPL before: {dev_ppl_before:.2f}  |  Test PPL before: {test_ppl_before:.2f}")

    # Setup MPT mechanisms
    subspace_bases = None
    metabolism = None
    mpt_cfg = cond.reg.mpt_config
    if mpt_cfg.subspace_protection_weight > 0:
        subspace_bases = principal_right_subspaces(
            brain._synapse.get_lora_weights(), mpt_cfg.subspace_rank,
        )
    if mpt_cfg.decay_mode not in (DM.NONE, DM.SALIENCY):
        metabolism = MetabolicLoop(
            base_lambda=mpt_cfg.decay_base_lambda,
            saliency_gamma=DECAY_GAMMA,
            mode=mpt_cfg.decay_mode,
            gershgorin_penalty=mpt_cfg.gershgorin_penalty,
        )

    # Training
    params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(params, lr=FEED_LR, weight_decay=cond.reg.weight_decay)

    t_train_start = time.perf_counter()
    total_steps = 0
    epoch_ppls: list[float] = []
    epoch_instability: list[float] = []
    epoch_diag_dominance: list[float] = []
    epoch_dev_ppls: list[float] = []
    epoch_shrink: list[float] = []
    best_dev_ppl = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(N_EPOCHS):
        random.shuffle(train_texts)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_texts), BATCH_SIZE):
            batch_texts = train_texts[i : i + BATCH_SIZE]
            batch, _ = brain._synapse.prepare_texts(batch_texts)
            model.train()
            optim.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            # Subspace protection — live parameters, so the term has a gradient
            if subspace_bases:
                sub_loss = subspace_leakage_loss(subspace_bases, brain._synapse.get_lora_parameters())
                loss = loss + mpt_cfg.subspace_protection_weight * sub_loss.to(loss.device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()

            # Spectral projection (post-step)
            if mpt_cfg.spectral_sparsity_retention < 1.0:
                spectral_projection_step(model, mpt_cfg.spectral_sparsity_retention)

            epoch_loss += loss.item()
            n_batches += 1
            total_steps += 1

        # Metabolic decay (end of epoch)
        if metabolism is not None:
            lora_w = brain._synapse.get_lora_weights()
            norm_before = merged_update_norm(lora_w)
            decayed = metabolism.tick(lora_w, delta_hours=DECAY_DELTA_HOURS)
            brain._synapse.set_lora_weights(decayed)
            norm_after = merged_update_norm(decayed)
            # Fractional shrink of ‖BA‖_F caused by this tick
            epoch_shrink.append(1.0 - norm_after / norm_before if norm_before > 0 else 0.0)

        avg = epoch_loss / max(n_batches, 1)
        ppl = math.exp(avg) if avg < 50 else float("inf")
        epoch_ppls.append(round(ppl, 4))

        # Dev PPL → early-stopping selection (test split is never looked at here)
        dev_ppl = compute_ppl(model, tokenizer, dev_texts)
        epoch_dev_ppls.append(round(dev_ppl, 4))
        if dev_ppl < best_dev_ppl:
            best_dev_ppl = dev_ppl
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in brain._synapse.get_lora_weights().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == N_EPOCHS - 1:
            print(f"    Epoch {epoch+1:>2}/{N_EPOCHS}: loss={avg:.4f}  train ppl={ppl:.2f}  dev ppl={dev_ppl:.2f}")

        # Per-epoch spectral trajectory (Phase 1.5 diagnostic)
        lora_w_epoch = brain._synapse.get_lora_weights()
        instabilities = gershgorin_lora_instability(lora_w_epoch)
        if instabilities:
            all_inst = torch.cat([v.view(-1) for v in instabilities.values()])
            epoch_instability.append(float(all_inst.mean().item()))
            # Diagonal-dominance ratio: fraction of rows where |diag| > R_i
            n_dominant = 0
            n_total = 0
            for key_a, A, key_b, B in _match_lora_pairs(lora_w_epoch):
                merged = B @ A
                if merged.shape[0] != merged.shape[1]:
                    continue
                radii = torch.sum(torch.abs(merged), dim=-1)
                diag = torch.abs(torch.diag(merged))
                dominant = (diag > radii - diag).float()
                n_dominant += int(dominant.sum().item())
                n_total += dominant.numel()
            epoch_diag_dominance.append(n_dominant / max(n_total, 1))
        else:
            epoch_instability.append(0.0)
            epoch_diag_dominance.append(1.0)

    t_train = time.perf_counter() - t_train_start

    # Final eval (last epoch), then early-stopped eval (best dev epoch)
    train_ppl = compute_ppl(model, tokenizer, train_texts)
    test_ppl_after = compute_ppl(model, tokenizer, test_texts)
    test_ppl_early = test_ppl_after
    if best_state is not None and best_epoch != N_EPOCHS:
        final_state = {k: v.clone() for k, v in brain._synapse.get_lora_weights().items()}
        brain._synapse.set_lora_weights(best_state)
        test_ppl_early = compute_ppl(model, tokenizer, test_texts)
        brain._synapse.set_lora_weights(final_state)

    # Effective rank
    from rengongtong._spectral import effective_rank_summary
    ranks = effective_rank_summary(model, 0.95)
    avg_rank = sum(ranks.values()) / len(ranks) if ranks else 0.0

    # DCT spectral energy (sampled)
    from rengongtong._spectral import lora_spectral_summary
    # use get_peft_model_state_dict or named_parameters
    lora_w = {}
    for name, p in model.named_parameters():
        if "lora_" in name and p.ndim == 2:
            lora_w[name] = p.data
    spectral_metrics = lora_spectral_summary(lora_w)
    # aggregate mean energy ratios across all layers
    energy_r10 = [v for k, v in spectral_metrics.items() if k.endswith("energy_r10%")]
    energy_mean_r10 = sum(energy_r10) / len(energy_r10) if energy_r10 else 0.0
    energy_r50 = [v for k, v in spectral_metrics.items() if k.endswith("energy_r50%")]
    energy_mean_r50 = sum(energy_r50) / len(energy_r50) if energy_r50 else 0.0

    # Determine which regularizer components are active
    active_components: list[str] = []
    if mpt_cfg.decay_mode in (DM.GERSHGORIN, DM.HYBRID, DM.ROTATED_GERSHGORIN):
        active_components.append("gershgorin_decay")
    if mpt_cfg.decay_mode == DM.DIAGONAL_MASS:
        active_components.append("diagonal_mass_decay")
    if mpt_cfg.decay_mode == DM.UNIFORM:
        active_components.append("uniform_shrink")
    if mpt_cfg.subspace_protection_weight > 0:
        active_components.append("subspace_protection")
    if mpt_cfg.decay_mode == DM.HYBRID:
        active_components.append("saliency_decay")
    if cond.reg.weight_decay > 0:
        active_components.append("weight_decay")
    if cond.reg.lora_dropout > 0:
        active_components.append("lora_dropout")

    result: dict[str, Any] = {
        "tag": cond.tag,
        "model": cond.model.name,
        "model_label": cond.model.label,
        "regularizer": cond.reg.tag,
        "regularizer_label": cond.reg.label,
        "seed": cond.seed,
        "load_time_s": round(t_load, 2),
        "train_time_s": round(t_train, 2),
        "total_steps": total_steps,
        "epoch_ppls": epoch_ppls,
        "epoch_instability": [round(v, 6) for v in epoch_instability],
        "epoch_diag_dominance": [round(v, 6) for v in epoch_diag_dominance],
        "epoch_dev_ppls": epoch_dev_ppls,
        "epoch_shrink": [round(v, 6) for v in epoch_shrink],
        "train_ppl": round(train_ppl, 4),
        "dev_ppl_before": round(dev_ppl_before, 4),
        "best_dev_ppl": round(best_dev_ppl, 4),
        "best_epoch": best_epoch,
        "test_ppl_before": round(test_ppl_before, 4),
        "test_ppl_after": round(test_ppl_after, 4),
        "test_ppl_early_stopped": round(test_ppl_early, 4),
        "delta_ppl": round(test_ppl_after - test_ppl_before, 4),
        "delta_ppl_early_stopped": round(test_ppl_early - test_ppl_before, 4),
        "avg_effective_rank": round(avg_rank, 2),
        "dct_energy_r10": round(energy_mean_r10, 4),
        "dct_energy_r50": round(energy_mean_r50, 4),
        "mpt_config": asdict(cond.reg.mpt_config),
        "lora_dropout": cond.reg.lora_dropout,
        "weight_decay": cond.reg.weight_decay,
        "regularizer_active_components": active_components,
        "optimizer_config": {"betas": (0.9, 0.999), "eps": 1e-8},
    }
    result.update(_pinned_metadata())

    print(f"  Train: {t_train:.1f}s ({total_steps} steps)")
    print(
        f"  Train PPL: {train_ppl:.2f}  |  Test: {test_ppl_before:.2f} → {test_ppl_after:.2f}  "
        f"(Δ={result['delta_ppl']:+.2f}; early-stopped @ epoch {best_epoch}: Δ={result['delta_ppl_early_stopped']:+.2f})"
    )
    print(f"  Avg eff rank: {avg_rank:.1f}  |  DCT energy@10%: {energy_mean_r10:.4f}")

    del brain
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _pinned_metadata() -> dict[str, Any]:
    """Capture reproducible-environment metadata."""
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


def save_commit_info():
    info = _pinned_metadata()
    (ROOT / "commit_info.json").write_text(json.dumps(info, indent=2))
    return info


def save_conditions(conds: list[Condition]):
    data = []
    i = 0
    for c in conds:
        data.append({
            "index": i,
            "tag": c.tag,
            "model": c.model.name,
            "regularizer": c.reg.tag,
            "seed": c.seed,
        })
        i += 1
    (ROOT / "conditions.json").write_text(json.dumps(data, indent=2))


def load_completed() -> set[str]:
    path = ROOT / "results_partial.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return {r["tag"] for r in data}


def save_result(result: dict):
    path = ROOT / "results_partial.json"
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = []
    # Replace if already exists
    data = [r for r in data if r["tag"] != result["tag"]]
    data.append(result)
    path.write_text(json.dumps(data, indent=2))
    print(f"  → Saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="experiment_002", help="results sub-directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models", nargs="+", default=None, help="e.g. 135M 1.7B")
    parser.add_argument("--regularizers", nargs="+", default=None, help="regularizer tags to run")
    return parser.parse_args()


def main():
    global ROOT
    args = parse_args()
    ROOT = _SCRIPT_DIR / "results" / args.experiment
    ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  EXPERIMENT: MPT spectral regularization for small LMs")
    print(f"  Root: {ROOT.resolve()}")
    print(f"  Matched shrink: weight_decay={MATCHED_WEIGHT_DECAY:.3f}  uniform λ₀={MATCHED_UNIFORM_LAMBDA0:.4f}")
    print("=" * 72)

    save_commit_info()
    train_texts, dev_texts, test_texts = load_wikitext()
    print(f"  Train: {len(train_texts)}  |  Dev: {len(dev_texts)}  |  Test: {len(test_texts)}")

    conditions = build_conditions(
        args.seeds,
        set(args.models) if args.models else None,
        set(args.regularizers) if args.regularizers else None,
    )
    save_conditions(conditions)
    print(f"  Total conditions: {len(conditions)}")

    completed = load_completed()
    print(f"  Already completed: {len(completed)}")

    results = []
    for i, cond in enumerate(conditions):
        if cond.tag in completed:
            print(f"\n  SKIP [{cond.tag}] — already completed")
            continue

        print(f"\n>>> [{i+1}/{len(conditions)}] {cond.tag}")
        try:
            result = train_condition(cond, train_texts, dev_texts, test_texts)
            results.append(result)
            save_result(result)
        except Exception as e:
            print(f"\n  ERROR [{cond.tag}]: {e}")
            import traceback
            traceback.print_exc()
            # Save error result to avoid re-running failing condition
            error_result = {
                "tag": cond.tag,
                "model": cond.model.name,
                "regularizer": cond.reg.tag,
                "seed": cond.seed,
                "error": str(e),
            }
            results.append(error_result)
            save_result(error_result)
        torch.cuda.empty_cache()

    # Build final report
    partial = json.loads((ROOT / "results_partial.json").read_text()) if (ROOT / "results_partial.json").exists() else []
    final = {
        "experiment": "mpt-small-lm-generalization",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_train": N_TRAIN,
            "n_dev": N_DEV,
            "n_test": N_TEST,
            "matched_shrink_per_epoch": MATCHED_SHRINK_PER_EPOCH,
            "matched_weight_decay": MATCHED_WEIGHT_DECAY,
            "matched_uniform_lambda0": MATCHED_UNIFORM_LAMBDA0,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": FEED_LR,
            "seeds": args.seeds,
        },
        "models": [m.name for m in MODELS],
        "regularizers": list({c.reg.tag for c in conditions}),
        "num_conditions": len(conditions),
        "num_completed": len(partial),
        "results": partial,
    }
    (ROOT / "report_paper.json").write_text(json.dumps(final, indent=2))
    print(f"\n{'=' * 72}")
    print(f"  Experiment complete. Report: {(ROOT / 'report_paper.json').resolve()}")
    print(f"  Completed: {len(partial)}/{len(conditions)} conditions")
    print(f"{'=' * 72}")

    # Summary table
    if partial:
        print_summary(partial)


def print_summary(results: list[dict]):
    grouped: dict[str, list[dict]] = {}
    for r in results:
        key = f"{r.get('model', '?')}/{r.get('regularizer', '?')}"
        grouped.setdefault(key, []).append(r)

    print("\n" + "=" * 72)
    print("  SUMMARY (mean ± std across seeds)")
    print("=" * 72)
    header = f"{'Model':<14} {'Reg':<22} {'Train PPL':>10} {'Test PPL':>10} {'ΔPPL':>8} {'ΔPPL(ES)':>9} {'EffRank':>8} {'E@10%':>8}"
    print(header)
    print("-" * len(header))
    for key in sorted(grouped.keys()):
        group = grouped[key]
        valid = [g for g in group if "error" not in g]
        if not valid:
            print(f"{'ERROR':>14} {key}")
            continue
        train_ppls = [g["train_ppl"] for g in valid]
        test_ppls = [g["test_ppl_after"] for g in valid]
        deltas = [g["delta_ppl"] for g in valid]
        deltas_es = [g["delta_ppl_early_stopped"] for g in valid]
        ranks = [g.get("avg_effective_rank", 0) for g in valid]
        energies = [g.get("dct_energy_r10", 0) for g in valid]
        parts = key.split("/")
        print(
            f"{parts[0]:<14} "
            f"{parts[1]:<22} "
            f"{sum(train_ppls)/len(train_ppls):>8.2f}±{_std(train_ppls):.2f} "
            f"{sum(test_ppls)/len(test_ppls):>8.2f}±{_std(test_ppls):.2f} "
            f"{sum(deltas)/len(deltas):>+7.2f}±{_std(deltas):.2f} "
            f"{sum(deltas_es)/len(deltas_es):>+7.2f}±{_std(deltas_es):.2f} "
            f"{sum(ranks)/len(ranks):>7.1f}±{_std(ranks):.1f} "
            f"{sum(energies)/len(energies):>7.4f}±{_std(energies):.4f}"
        )
    print("=" * 72)


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


if __name__ == "__main__":
    main()
