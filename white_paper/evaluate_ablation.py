#!/usr/bin/env python3
"""Ablation study: compare Vanilla LoRA vs Weight Decay vs EWC vs MPT."""

from __future__ import annotations

import os
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Ensure rengongtong package is accessible
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from rengongtong.brain import Brain
from rengongtong.synaptic import BASE_MODEL, SynapticManager
from rengongtong._spectral import (
    gershgorin_lora_instability,
    pseudospectral_sensitivity,
    subspace_rotation_loss,
    compute_saliency,
)
from rengongtong._types import MptConfig, DecayMode
from rengongtong._ewc import ElasticWeightConsolidation

# ---------------------------------------------------------------------------
# Ansbach test data
# ---------------------------------------------------------------------------

ANSBACH_TEXT = """Ansbach ist eine kreisfreie Stadt in Bayern und zählt zur Planungsregion Westmittelfranken und Metropolregion Nürnberg. Sie ist Sitz der Regierung und der Bezirksverwaltung von Mittelfranken sowie des Landratsamtes Ansbach. Die Stadt liegt etwa 40 Kilometer südwestlich von Nürnberg am Zufluss des Onolzbachs in die Fränkische Rezat, die letztlich zum Main entwässert. Sie ist nach Fläche hinter München, Nürnberg, Augsburg und Ingolstadt die fünftgrößte kreisfreie Stadt des Freistaates Bayern.

Im Jahre 748 wurde im Mündungswinkel des Onoldsbaches zur Rezat vom fränkischen Edelfreien Gumbert ein Benediktinerkloster gegründet; vom heute meist Onolzbach geschriebenen Rezatzufluss ist der spätere Name Ansbach abgeleitet. In den folgenden Jahrhunderten wuchsen das Kloster und die daneben liegende Siedlung zu einer Stadt zusammen. 1221 wurde der Ort das erste Mal als Stadt erwähnt.

Ab 1385 bis 1791 war Ansbach die Haupt- und Residenzstadt verschiedener zollerscher Herrschaftsbereiche. 1791 verzichtete der letzte Markgraf Karl Alexander von Brandenburg-Ansbach gegen eine jährliche Leibrente auf sein Herrschaftsgebiet und trat seine beiden Fürstentümer Ansbach und Bayreuth an Preußen ab.

1796 wählte Maximilian Joseph, Herzog von Zweibrücken und bayerischer Kurprätendent, Ansbach zu seiner Exilresidenz. Maximilian von Montgelas entwickelte dort das Ansbacher Memoire, ein umfassendes Konzept einer künftigen radikalen politischen Neugestaltung Bayerns. Nach 1806 fiel Ansbach an das Königreich Bayern und wurde Hauptstadt des Rezatkreises, aus dem 1838 Mittelfranken wurde.

Ansbach ist bekannt für die Residenz der Markgrafen zu Brandenburg-Ansbach mit ihrer Sammlung von Fayencen und Porzellan aus der ehemaligen Ansbacher Manufaktur. Die Gumbertuskirche besitzt die größte Barockorgel im fränkischen Raum. Das Gymnasium Carolinum wurde 1528 gegründet und ist das zweitälteste nichtklösterliche Gymnasium Bayerns.

Heute hat Ansbach rund 40.000 Einwohner. Die Stadt gliedert sich in 54 Gemeindeteile. Ansbach pflegt Städtepartnerschaften mit Bay City in den USA, Anglet in Frankreich und Fermo in Italien."""


# ---------------------------------------------------------------------------
# Variant definitions
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
        mpt_config=MptConfig(
            decay_mode=DecayMode.SALIENCY,
            subspace_protection_weight=0.0,
        ),
        optim_weight_decay=0.0,
    ),
    AblationVariant(
        tag="weight_decay",
        label="LoRA + AdamW weight_decay=0.01",
        mpt_config=MptConfig(
            decay_mode=DecayMode.SALIENCY,
            subspace_protection_weight=0.0,
        ),
        optim_weight_decay=0.01,
    ),
    AblationVariant(
        tag="ewc",
        label="LoRA + EWC (λ=0.1)",
        mpt_config=MptConfig(
            decay_mode=DecayMode.SALIENCY,
            subspace_protection_weight=0.0,
        ),
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
# Metrics collector
# ---------------------------------------------------------------------------


def collect_metrics(
    brain: Brain, input_ids: torch.Tensor, attn_mask: torch.Tensor,
    tokenizer, base_weights: dict | None = None,
) -> dict:
    model = brain.model
    model.eval()

    with torch.inference_mode():
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss.item()
    ppl = math.exp(loss) if loss < 50 else float("inf")

    with torch.inference_mode():
        gap = pseudospectral_sensitivity(
            forward_fn=model,
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
            epsilon=brain.mpt.pseudospectral_epsilon,
        )
    stability_gap = gap.item()

    lora_weights = brain._synapse.get_lora_weights()
    gersh = gershgorin_lora_instability(lora_weights)
    if gersh:
        gersh_mean = sum(v.mean().item() for v in gersh.values()) / len(gersh)
        gersh_max = max(v.max().item() for v in gersh.values())
    else:
        gersh_mean = 0.0
        gersh_max = 0.0

    if base_weights is not None:
        subspace = subspace_rotation_loss(base_weights, lora_weights, rank=brain.mpt.subspace_rank).item()
    else:
        subspace = 0.0

    saliency = compute_saliency(lora_weights)
    sal_vals = torch.cat([v.flatten() for v in saliency.values()])
    sample = sal_vals[torch.randperm(len(sal_vals), device=sal_vals.device)[:10000]]
    sal_mean = sal_vals.mean().item()

    return {
        "loss": round(loss, 4),
        "perplexity": round(ppl, 2),
        "stability_gap": round(stability_gap, 6),
        "gershgorin_instability_mean": round(gersh_mean, 6),
        "gershgorin_instability_max": round(gersh_max, 6),
        "subspace_rotation_loss": round(subspace, 6),
        "saliency_mean": round(sal_mean, 4),
    }


def run_variant(variant: AblationVariant) -> dict:
    print(f"\n{'=' * 60}")
    print(f"[{variant.tag}] {variant.label}")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    brain = Brain(
        model_name=BASE_MODEL,
        mpt=variant.mpt_config,
    )
    model = brain.model
    tokenizer = brain.tokenizer
    t_load = time.perf_counter() - t0

    # Tokenize Ansbach text for metrics
    inputs = tokenizer(ANSBACH_TEXT, truncation=True, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attn_mask = inputs.attention_mask.to(model.device)

    # Baseline metrics
    base_weights = {k: v.clone() for k, v in brain._synapse.get_lora_weights().items()}
    pre_metrics = collect_metrics(brain, input_ids, attn_mask, tokenizer, base_weights=base_weights)
    print(f"  Load: {t_load:.1f}s  |  Pre PPL: {pre_metrics['perplexity']}  "
          f"Stab: {pre_metrics['stability_gap']:.4f}  "
          f"Gersh μ: {pre_metrics['gershgorin_instability_mean']:.4f}")

    # Feed Ansbach text
    # For EWC: compute Fisher first, then feed with EWC penalty
    ewc_controller = None
    custom_loss_fn = None
    if variant.use_ewc:
        ewc_controller = ElasticWeightConsolidation(weight=0.1)
        ewc_controller.compute_fisher(
            model, input_ids, attn_mask,
            labels=input_ids.clone(), n_samples=3,
        )
        def make_ewc_fn(ctrl):
            def fn(m, b, o):
                return ctrl.penalty(m)
            return fn
        custom_loss_fn = make_ewc_fn(ewc_controller)

    t_train_start = time.perf_counter()
    report = brain.feed(
        ANSBACH_TEXT,
        steps=3,
        custom_loss_fn=custom_loss_fn,
    )
    t_train = time.perf_counter() - t_train_start

    # Post-feed metrics
    post_metrics = collect_metrics(brain, input_ids, attn_mask, tokenizer, base_weights=base_weights)
    print(f"  Train: {t_train:.2f}s  |  Loss: {report.loss:.4f}  "
          f"PPL: {pre_metrics['perplexity']}→{post_metrics['perplexity']}  "
          f"Subspace: {post_metrics['subspace_rotation_loss']:.4f}")

    result = {
        "variant": variant.tag,
        "label": variant.label,
        "mpt_config": asdict(variant.mpt_config),
        "optim_weight_decay": variant.optim_weight_decay,
        "use_ewc": variant.use_ewc,
        "load_time_s": round(t_load, 2),
        "train_time_s": round(t_train, 2),
        "train_loss": round(report.loss, 4),
        "baseline": pre_metrics,
        "post_feed": post_metrics,
    }

    # PPL delta
    ppl_before = pre_metrics["perplexity"]
    ppl_after = post_metrics["perplexity"]
    if ppl_before and ppl_before != float("inf"):
        result["ppl_delta"] = round(ppl_before - ppl_after, 2)
        result["ppl_improvement_pct"] = round((ppl_before - ppl_after) / ppl_before * 100, 2)
    else:
        result["ppl_delta"] = 0.0
        result["ppl_improvement_pct"] = 0.0

    del brain
    torch.cuda.empty_cache()

    return result


def main() -> None:
    print("=" * 72)
    print("ABLATION STUDY — SmolLM2-1.7B on Ansbach data")
    print("=" * 72)

    results = []
    for variant in VARIANTS:
        try:
            result = run_variant(variant)
            results.append(result)
        except Exception as e:
            print(f"  ERROR [{variant.tag}]: {e}")
            results.append({
                "variant": variant.tag,
                "label": variant.label,
                "error": str(e),
            })
        torch.cuda.empty_cache()

    # Save results
    out_path = _SCRIPT_DIR / "ablation_report.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path.resolve()}")

    # Summary table
    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    header = f"{'Variant':<20} {'PPL ↓':>8} {'ΔPPL':>8} {'Stab':>8} {'Gersh μ':>10} {'Subsp':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['variant']:<20} {'ERROR':>8} {'':>8} {'':>8} {'':>10} {'':>8}")
            continue
        post = r["post_feed"]
        print(
            f"{r['variant']:<20} "
            f"{post['perplexity']:>8.2f} "
            f"{r['ppl_delta']:>+8.2f} "
            f"{post['stability_gap']:>8.4f} "
            f"{post['gershgorin_instability_mean']:>10.6f} "
            f"{post['subspace_rotation_loss']:>8.4f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
