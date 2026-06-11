#!/usr/bin/env python3
"""Systematic experiment: MPT spectral regularization for small LMs.

Design:
  A — Scale × Regularizer (135M, 360M, 1.7B) × (vanilla, weight_decay, mpt_full) × 3 seeds
  B — MPT component ablation (1.7B) × (mpt_gershgorin, mpt_subspace) × 3 seeds
  C — LoRA dropout baseline (1.7B) × (dropout) × 3 seeds

Total: 27 + 6 + 3 = 36 training runs

Expected wall time: ~2.5 hours on RTX A5000.
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
from rengongtong._spectral import gershgorin_lora_instability, _match_lora_pairs, subspace_rotation_loss, spectral_projection_step
from rengongtong.metabolism import MetabolicLoop

# ---------------------------------------------------------------------------
# Experiment root (relative to this script's location)
# ---------------------------------------------------------------------------

ROOT = _SCRIPT_DIR / "results" / "experiment_001"
ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

N_TRAIN = 100
N_VAL = 500
BATCH_SIZE = 8
N_EPOCHS = 15
FEED_LR = 5e-5
SEEDS = [42, 73, 137]

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


def _mpt(decay_mode: str, subspace_w: float) -> MptConfig:
    decay_penalty = 0.1 if decay_mode in (
        DM.GERSHGORIN, DM.HYBRID, DM.DIAGONAL_MASS, DM.ROTATED_GERSHGORIN,
    ) else 0.0
    return MptConfig(
        decay_mode=decay_mode,
        gershgorin_penalty=decay_penalty,
        pseudospectral_epsilon=1e-5,
        pseudospectral_threshold=10.0,
        subspace_protection_weight=subspace_w,
        subspace_rank=8,
    )


VANILLA   = RegularizerDef("vanilla",         "Vanilla LoRA",              MptConfig(decay_mode=DM.NONE, subspace_protection_weight=0.0))
WEIGHT_DECAY = RegularizerDef("weight_decay", "AdamW weight_decay=0.01",   MptConfig(decay_mode=DM.NONE, subspace_protection_weight=0.0), weight_decay=0.01)
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


def build_conditions() -> list[Condition]:
    conds = []
    # A: Scale × Regularizer
    for m in MODELS:
        for r in [VANILLA, WEIGHT_DECAY, MPT_FULL]:
            for s in SEEDS:
                conds.append(Condition(m, r, s))
    # B: MPT component ablation (1.7B only)
    m17 = MODELS[2]
    for r in [MPT_GERSH, MPT_SUB]:
        for s in SEEDS:
            conds.append(Condition(m17, r, s))
    # C: Dropout baseline (1.7B only)
    for s in SEEDS:
        conds.append(Condition(m17, DROPOUT, s))
    return conds


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_wikitext() -> tuple[list[str], list[str]]:
    print("Loading Wikitext-2...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    all_train = [t for t in ds["train"]["text"] if t.strip()]
    all_val = [t for t in ds["validation"]["text"] if t.strip()]
    return all_train[:N_TRAIN], all_val[:N_VAL]


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

def train_condition(cond: Condition, train_texts: list[str], val_texts: list[str]) -> dict[str, Any]:
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
    t_eval_start = time.perf_counter()
    val_ppl_before = compute_ppl(model, tokenizer, val_texts)
    t_eval = time.perf_counter() - t_eval_start
    print(f"  Load: {t_load:.1f}s  |  Val PPL before: {val_ppl_before:.2f}")

    # Setup MPT mechanisms
    base_weights = None
    metabolism = None
    mpt_cfg = cond.reg.mpt_config
    if mpt_cfg.subspace_protection_weight > 0:
        base_weights = brain._synapse.get_lora_weights()
    if mpt_cfg.decay_mode in (DM.GERSHGORIN, DM.HYBRID, DM.DIAGONAL_MASS, DM.ROTATED_GERSHGORIN):
        metabolism = MetabolicLoop(
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

            # Subspace protection
            if base_weights is not None:
                current = brain._synapse.get_lora_weights()
                sub_loss = subspace_rotation_loss(base_weights, current, mpt_cfg.subspace_rank)
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
            decayed = metabolism.tick(lora_w, delta_hours=1.0)
            brain._synapse.set_lora_weights(decayed)

        avg = epoch_loss / max(n_batches, 1)
        ppl = math.exp(avg) if avg < 50 else float("inf")
        epoch_ppls.append(round(ppl, 4))
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == N_EPOCHS - 1:
            print(f"    Epoch {epoch+1:>2}/{N_EPOCHS}: loss={avg:.4f}  ppl={ppl:.2f}")

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

    # Final eval
    t_eval2_start = time.perf_counter()
    train_ppl = compute_ppl(model, tokenizer, train_texts)
    val_ppl_after = compute_ppl(model, tokenizer, val_texts)
    t_eval2 = time.perf_counter() - t_eval2_start

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
        "train_ppl": round(train_ppl, 4),
        "val_ppl_before": round(val_ppl_before, 4),
        "val_ppl_after": round(val_ppl_after, 4),
        "delta_ppl": round(val_ppl_after - val_ppl_before, 4),
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
    print(f"  Train PPL: {train_ppl:.2f}  |  Val: {val_ppl_before:.2f} → {val_ppl_after:.2f}  (Δ={result['delta_ppl']:+.2f})")
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

def main():
    print("=" * 72)
    print("  EXPERIMENT: MPT spectral regularization for small LMs")
    print(f"  Root: {ROOT.resolve()}")
    print("=" * 72)

    save_commit_info()
    train_texts, val_texts = load_wikitext()
    print(f"  Train: {len(train_texts)}  |  Val: {len(val_texts)}")

    conditions = build_conditions()
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
            result = train_condition(cond, train_texts, val_texts)
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
            "n_val": N_VAL,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": FEED_LR,
            "seeds": SEEDS,
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
    header = f"{'Model':<14} {'Reg':<16} {'Train PPL':>10} {'Val PPL':>10} {'ΔPPL':>8} {'EffRank':>8} {'E@10%':>8}"
    print(header)
    print("-" * len(header))
    for key in sorted(grouped.keys()):
        group = grouped[key]
        valid = [g for g in group if "error" not in g]
        if not valid:
            print(f"{'ERROR':>14} {key}")
            continue
        train_ppls = [g["train_ppl"] for g in valid]
        val_ppls = [g["val_ppl_after"] for g in valid]
        deltas = [g["delta_ppl"] for g in valid]
        ranks = [g.get("avg_effective_rank", 0) for g in valid]
        energies = [g.get("dct_energy_r10", 0) for g in valid]
        parts = key.split("/")
        print(
            f"{parts[0]:<14} "
            f"{parts[1]:<16} "
            f"{sum(train_ppls)/len(train_ppls):>8.2f}±{_std(train_ppls):.2f} "
            f"{sum(val_ppls)/len(val_ppls):>8.2f}±{_std(val_ppls):.2f} "
            f"{sum(deltas)/len(deltas):>+7.2f}±{_std(deltas):.2f} "
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
