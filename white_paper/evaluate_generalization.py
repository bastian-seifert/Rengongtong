#!/usr/bin/env python3
"""Train 4 variants on Wikitext-2 (100 train) and measure held-out PPL (500 val)."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Ensure rengongtong package is accessible
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
from rengongtong._types import MptConfig, DecayMode
from rengongtong._ewc import ElasticWeightConsolidation

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

N_TRAIN = 100
N_VAL = 500
BATCH_SIZE = 8
N_EPOCHS = 15
FEED_LR = 5e-5

# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


@dataclass
class AblationVariant:
    tag: str
    label: str
    mpt_config: MptConfig
    optim_weight_decay: float = 0.0
    use_ewc: bool = False


VARIANTS = [
    AblationVariant(
        tag="vanilla",
        label="Vanilla LoRA (no decay)",
        mpt_config=MptConfig(decay_mode=DecayMode.SALIENCY, subspace_protection_weight=0.0),
        optim_weight_decay=0.0,
    ),
    AblationVariant(
        tag="weight_decay",
        label="LoRA + AdamW weight_decay=0.01",
        mpt_config=MptConfig(decay_mode=DecayMode.SALIENCY, subspace_protection_weight=0.0),
        optim_weight_decay=0.01,
    ),
    AblationVariant(
        tag="ewc",
        label="LoRA + EWC (λ=0.1)",
        mpt_config=MptConfig(decay_mode=DecayMode.SALIENCY, subspace_protection_weight=0.0),
        use_ewc=True,
    ),
    AblationVariant(
        tag="mpt",
        label="LoRA + MPT (hybrid + subspace)",
        mpt_config=MptConfig(
            decay_mode=DecayMode.HYBRID,
            gershgorin_penalty=0.1,
            pseudospectral_epsilon=1e-5,
            pseudospectral_threshold=10.0,
            subspace_protection_weight=0.05,
            subspace_rank=8,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_ppl(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
) -> float:
    """Compute average perplexity over *texts*, batching for efficiency."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tok = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt",
            )
            input_ids = tok.input_ids.to(model.device)
            attn_mask = tok.attention_mask.to(model.device)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss.item()
            n = (labels != -100).sum().item()
            total_loss += loss * n
            total_tokens += n
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss) if avg_loss < 50 else float("inf")


def train_variant(
    variant: AblationVariant,
    train_texts: list[str],
    val_texts: list[str],
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"[{variant.tag}] {variant.label}")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    brain = Brain(model_name=BASE_MODEL, mpt=variant.mpt_config)
    model = brain.model
    tokenizer = brain.tokenizer
    t_load = time.perf_counter() - t0

    # Baseline PPL
    t1 = time.perf_counter()
    val_ppl_before = compute_ppl(model, tokenizer, val_texts)
    t_eval = time.perf_counter() - t1
    print(f"  Load: {t_load:.1f}s  |  Val PPL before: {val_ppl_before:.2f}  (eval: {t_eval:.1f}s)")

    # EWC setup
    ewc_controller = None
    custom_loss_fn = None
    if variant.use_ewc:
        # Compute Fisher on first train batch
        tok = tokenizer(
            train_texts[:8], truncation=True, padding=True, max_length=2048, return_tensors="pt",
        )
        ewc_controller = ElasticWeightConsolidation(weight=0.1)
        ewc_controller.compute_fisher(
            model,
            tok.input_ids.to(model.device),
            tok.attention_mask.to(model.device),
            labels=tok.input_ids.to(model.device).clone(),
            n_samples=3,
        )

        def make_ewc_fn(ctrl):
            def fn(m, b, o):
                return ctrl.penalty(m)
            return fn
        custom_loss_fn = make_ewc_fn(ewc_controller)

    # Training loop (persistent optimizer across all batches/epochs)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(params, lr=FEED_LR, weight_decay=variant.optim_weight_decay)

    t_train_start = time.perf_counter()
    total_steps = 0
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
            if custom_loss_fn is not None:
                extra = custom_loss_fn(model, batch, outputs)
                if extra is not None:
                    loss = loss + extra
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()

            # Spectral projection (if configured)
            retention = variant.mpt_config.spectral_sparsity_retention
            if retention < 1.0:
                from rengongtong._spectral import spectral_projection_step
                spectral_projection_step(model, retention)

            epoch_loss += loss.item()
            n_batches += 1
            total_steps += 1

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{N_EPOCHS}: avg_loss={avg_epoch_loss:.4f}  "
              f"ppl={math.exp(avg_epoch_loss) if avg_epoch_loss < 50 else float('inf'):.2f}")

    t_train = time.perf_counter() - t_train_start

    # Held-out evaluation
    t2 = time.perf_counter()
    val_ppl_after = compute_ppl(model, tokenizer, val_texts)
    t_eval2 = time.perf_counter() - t2

    # Train PPL (last epoch)
    train_ppl = math.exp(avg_epoch_loss) if avg_epoch_loss < 50 else float("inf")

    result = {
        "variant": variant.tag,
        "label": variant.label,
        "mpt_config": asdict(variant.mpt_config),
        "load_time_s": round(t_load, 2),
        "train_time_s": round(t_train, 2),
        "eval_time_s": round(t_eval + t_eval2, 2),
        "total_steps": total_steps,
        "train_ppl": round(train_ppl, 2),
        "val_ppl_before": round(val_ppl_before, 2),
        "val_ppl_after": round(val_ppl_after, 2),
        "delta_ppl": round(val_ppl_after - val_ppl_before, 2),
    }

    print(f"  Train: {t_train:.1f}s ({total_steps} steps)  |  "
          f"Train PPL: {train_ppl:.2f}  |  "
          f"Val: {val_ppl_before:.2f} → {val_ppl_after:.2f}  "
          f"(Δ={result['delta_ppl']:+.2f})")

    del brain
    torch.cuda.empty_cache()
    return result


def main() -> None:
    print("=" * 72)
    print("GENERALIZATION STUDY — SmolLM2-1.7B on Wikitext-2")
    print("=" * 72)

    # Load dataset
    print("\nLoading Wikitext-2...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    all_train = [t for t in ds["train"]["text"] if t.strip()]
    all_val = [t for t in ds["validation"]["text"] if t.strip()]
    train_texts = all_train[:N_TRAIN]
    val_texts = all_val[:N_VAL]
    print(f"  Train: {len(train_texts)}  |  Val: {len(val_texts)}")

    results = []
    for variant in VARIANTS:
        try:
            result = train_variant(variant, train_texts, val_texts)
            results.append(result)
        except Exception as e:
            print(f"  ERROR [{variant.tag}]: {e}")
            import traceback
            traceback.print_exc()
            results.append({"variant": variant.tag, "label": variant.label, "error": str(e)})
        torch.cuda.empty_cache()

    # Save
    out_path = _SCRIPT_DIR / "generalization_report.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nReport saved to {out_path.resolve()}")

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    col_w = 15
    header = f"{'Variant':<{col_w}} {'Train PPL':>10} {'Val Before':>12} {'Val After':>10} {'ΔPPL':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['variant']:<{col_w}} {'ERROR':>10}")
            continue
        print(
            f"{r['variant']:<{col_w}} "
            f"{r['train_ppl']:>10.2f} "
            f"{r['val_ppl_before']:>12.2f} "
            f"{r['val_ppl_after']:>10.2f} "
            f"{r['delta_ppl']:>+8.2f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
