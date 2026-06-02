#!/usr/bin/env python3
"""Evaluate MPT-enhanced LoRA quality on SmolLM2-1.7B using Ansbach data."""

from __future__ import annotations

import os
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import json
import math
import sys
import time
from pathlib import Path

# Ensure rengongtong package is accessible
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from rengongtong.brain import Brain
from rengongtong.synaptic import SynapticManager, BASE_MODEL
from rengongtong._spectral import (
    gershgorin_lora_instability,
    pseudospectral_sensitivity,
    subspace_rotation_loss,
    compute_saliency,
)
from rengongtong._types import MptConfig, DecayMode

# ---------------------------------------------------------------------------
# 1.  Ansbach Wikipedia article (cleaned text)
# ---------------------------------------------------------------------------

ANSBACH_TEXT = """Ansbach ist eine kreisfreie Stadt in Bayern und zählt zur Planungsregion Westmittelfranken und Metropolregion Nürnberg. Sie ist Sitz der Regierung und der Bezirksverwaltung von Mittelfranken sowie des Landratsamtes Ansbach. Die Stadt liegt etwa 40 Kilometer südwestlich von Nürnberg am Zufluss des Onolzbachs in die Fränkische Rezat, die letztlich zum Main entwässert. Sie ist nach Fläche hinter München, Nürnberg, Augsburg und Ingolstadt die fünftgrößte kreisfreie Stadt des Freistaates Bayern.

Im Jahre 748 wurde im Mündungswinkel des Onoldsbaches zur Rezat vom fränkischen Edelfreien Gumbert ein Benediktinerkloster gegründet; vom heute meist Onolzbach geschriebenen Rezatzufluss ist der spätere Name Ansbach abgeleitet. In den folgenden Jahrhunderten wuchsen das Kloster und die daneben liegende Siedlung zu einer Stadt zusammen. 1221 wurde der Ort das erste Mal als Stadt erwähnt.

Ab 1385 bis 1791 war Ansbach die Haupt- und Residenzstadt verschiedener zollerscher Herrschaftsbereiche. 1791 verzichtete der letzte Markgraf Karl Alexander von Brandenburg-Ansbach gegen eine jährliche Leibrente auf sein Herrschaftsgebiet und trat seine beiden Fürstentümer Ansbach und Bayreuth an Preußen ab.

1796 wählte Maximilian Joseph, Herzog von Zweibrücken und bayerischer Kurprätendent, Ansbach zu seiner Exilresidenz. Maximilian von Montgelas entwickelte dort das Ansbacher Memoire, ein umfassendes Konzept einer künftigen radikalen politischen Neugestaltung Bayerns. Nach 1806 fiel Ansbach an das Königreich Bayern und wurde Hauptstadt des Rezatkreises, aus dem 1838 Mittelfranken wurde.

Ansbach ist bekannt für die Residenz der Markgrafen zu Brandenburg-Ansbach mit ihrer Sammlung von Fayencen und Porzellan aus der ehemaligen Ansbacher Manufaktur. Die Gumbertuskirche besitzt die größte Barockorgel im fränkischen Raum. Das Gymnasium Carolinum wurde 1528 gegründet und ist das zweitälteste nichtklösterliche Gymnasium Bayerns.

Heute hat Ansbach rund 40.000 Einwohner. Die Stadt gliedert sich in 54 Gemeindeteile. Ansbach pflegt Städtepartnerschaften mit Bay City in den USA, Anglet in Frankreich und Fermo in Italien."""


def main() -> None:
    print("=" * 72)
    print("MPT QUALITY EVALUATION — SmolLM2-1.7B on Ansbach data")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 2.  Load model (this will download SmolLM2-1.7B if not cached)
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading {BASE_MODEL} ...")
    t0 = time.perf_counter()

    mpt_cfg = MptConfig(
        decay_mode=DecayMode.HYBRID,
        gershgorin_penalty=0.1,
        pseudospectral_epsilon=1e-5,
        pseudospectral_threshold=10.0,
        subspace_protection_weight=0.05,
        subspace_rank=8,
    )
    brain = Brain(model_name=BASE_MODEL, mpt=mpt_cfg)
    synapse = brain._synapse
    model = brain.model
    tokenizer = brain.tokenizer

    t_load = time.perf_counter() - t0
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Loaded in {t_load:.1f}s — {param_count/1e6:.0f}M total params, {trainable/1e3:.0f}K trainable")
    print(f"   Device: {model.device}")

    # Save model info
    model_info = {
        "model": BASE_MODEL,
        "device": str(model.device),
        "param_count": param_count,
        "trainable_params": trainable,
        "load_time_s": round(t_load, 2),
    }

    # ------------------------------------------------------------------
    # 3.  Baseline metrics (before feeding)
    # ------------------------------------------------------------------
    print(f"\n[2/6] Baseline MPT metrics (pre-feed) ...")
    torch.inference_mode()

    # Perplexity on Ansbach text
    inputs = tokenizer(ANSBACH_TEXT, truncation=True, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attn_mask = inputs.attention_mask.to(model.device)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    model.eval()

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        baseline_loss = outputs.loss.item()
    baseline_ppl = math.exp(baseline_loss) if baseline_loss < 50 else float("inf")
    print(f"   Baseline perplexity: {baseline_ppl:.2f}")

    # Pseudospectral stability
    with torch.inference_mode():
        gap = pseudospectral_sensitivity(
            forward_fn=model,
            input_ids=input_ids,
            attention_mask=attn_mask,
            epsilon=mpt_cfg.pseudospectral_epsilon,
        )
    baseline_stability_gap = gap.item()
    print(f"   Baseline stability gap: {baseline_stability_gap:.6f}")

    # Gershgorin instability of initial LoRA (fresh init, should be near zero)
    lora_weights = synapse.get_lora_weights()
    gersh_instability = gershgorin_lora_instability(lora_weights)
    if gersh_instability:
        avg_instability = sum(v.mean().item() for v in gersh_instability.values()) / len(gersh_instability)
        max_instability = max(v.max().item() for v in gersh_instability.values())
    else:
        avg_instability = 0.0
        max_instability = 0.0
    print(f"   Gershgorin instability (mean): {avg_instability:.6f}")
    print(f"   Gershgorin instability (max):  {max_instability:.6f}")

    # Subspace rotation loss (baseline = current vs. current = 0)
    base_weights = {k: v.clone() for k, v in lora_weights.items()}
    subspace_loss = subspace_rotation_loss(base_weights, lora_weights, rank=mpt_cfg.subspace_rank)
    print(f"   Subspace rotation loss: {subspace_loss.item():.6f}")

    # Saliency stats
    saliency = compute_saliency(lora_weights)
    saliency_values = torch.cat([v.flatten() for v in saliency.values()])
    # Sample to compute quantiles (avoid OOM on large tensors)
    sample = saliency_values[torch.randperm(len(saliency_values))[:10000]]
    print(f"   Saliency — mean={saliency_values.mean().item():.4f}  "
          f"median={sample.median().item():.4f}  "
          f"p90={sample.quantile(0.9).item():.4f}")

    baseline_report = {
        "perplexity": round(baseline_ppl, 2),
        "stability_gap": round(baseline_stability_gap, 6),
        "gershgorin_instability_mean": round(avg_instability, 6),
        "gershgorin_instability_max": round(max_instability, 6),
        "subspace_rotation_loss": round(subspace_loss.item(), 6),
        "saliency_mean": round(saliency_values.mean().item(), 4),
        "saliency_median": round(sample.median().item(), 4),
        "saliency_p90": round(sample.quantile(0.9).item(), 4),
    }

    # ------------------------------------------------------------------
    # 4.  Feed Ansbach text (train the LoRA)
    # ------------------------------------------------------------------
    print(f"\n[3/6] Feeding Ansbach data ({len(ANSBACH_TEXT)} chars) ...")
    t0 = time.perf_counter()
    report = brain.feed(ANSBACH_TEXT, steps=3)
    t_train = time.perf_counter() - t0
    print(f"   Loss: {report.loss:.4f}  —  PPL after: {report.perplexity_after:.2f}")
    print(f"   Duration: {report.duration_seconds:.2f}s")

    training_report = {
        "steps": report.steps,
        "loss": round(report.loss, 4),
        "perplexity_after": round(report.perplexity_after, 2),
        "duration_s": round(report.duration_seconds, 2),
    }

    # ------------------------------------------------------------------
    # 5.  Post-feed MPT metrics
    # ------------------------------------------------------------------
    print(f"\n[4/6] Post-feed MPT metrics ...")
    model.eval()

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        post_loss = outputs.loss.item()
    post_ppl = math.exp(post_loss) if post_loss < 50 else float("inf")
    print(f"   Post-feed perplexity: {post_ppl:.2f}")
    ppl_delta = baseline_ppl - post_ppl if post_ppl != float("inf") else 0
    print(f"   PPL reduction: {ppl_delta:.2f} ({ppl_delta/baseline_ppl*100:.1f}% improvement)" if baseline_ppl > 0 else "")

    with torch.inference_mode():
        gap = pseudospectral_sensitivity(
            forward_fn=model,
            input_ids=input_ids,
            attention_mask=attn_mask,
            epsilon=mpt_cfg.pseudospectral_epsilon,
        )
    post_stability_gap = gap.item()
    print(f"   Post-feed stability gap: {post_stability_gap:.6f}  "
          f"(delta: {post_stability_gap - baseline_stability_gap:+.6f})")

    lora_weights = synapse.get_lora_weights()
    gersh_instability = gershgorin_lora_instability(lora_weights)
    if gersh_instability:
        avg_instability = sum(v.mean().item() for v in gersh_instability.values()) / len(gersh_instability)
        max_instability = max(v.max().item() for v in gersh_instability.values())
    else:
        avg_instability = 0.0
        max_instability = 0.0
    print(f"   Gershgorin instability (mean): {avg_instability:.6f}")
    print(f"   Gershgorin instability (max):  {max_instability:.6f}")

    subspace_loss = subspace_rotation_loss(base_weights, lora_weights, rank=mpt_cfg.subspace_rank)
    print(f"   Subspace rotation loss: {subspace_loss.item():.6f}")

    saliency = compute_saliency(lora_weights)
    saliency_values = torch.cat([v.flatten() for v in saliency.values()])
    sample = saliency_values[torch.randperm(len(saliency_values))[:10000]]
    print(f"   Saliency — mean={saliency_values.mean().item():.4f}  "
          f"median={sample.median().item():.4f}  "
          f"p90={sample.quantile(0.9).item():.4f}")

    post_report = {
        "perplexity": round(post_ppl, 2),
        "perplexity_delta": round(-ppl_delta, 2),
        "stability_gap": round(post_stability_gap, 6),
        "stability_gap_delta": round(post_stability_gap - baseline_stability_gap, 6),
        "gershgorin_instability_mean": round(avg_instability, 6),
        "gershgorin_instability_max": round(max_instability, 6),
        "subspace_rotation_loss": round(subspace_loss.item(), 6),
        "saliency_mean": round(saliency_values.mean().item(), 4),
        "saliency_median": round(sample.median().item(), 4),
        "saliency_p90": round(sample.quantile(0.9).item(), 4),
    }

    # ------------------------------------------------------------------
    # 6.  Generation test — chat about Ansbach
    # ------------------------------------------------------------------
    print(f"\n[5/6] Generation test — chatting about Ansbach ...")
    questions = [
        "Was ist Ansbach?",
        "Wann wurde Ansbach gegründet?",
        "Was ist das Ansbacher Memoire?",
        "Welche Sehenswürdigkeiten gibt es in Ansbach?",
    ]
    generations = {}
    for q in questions:
        response = brain.chat(q, max_new_tokens=128, temperature=0.7)
        print(f"\n   Q: {q}")
        print(f"   A: {response[:200]}{'...' if len(response) > 200 else ''}")
        generations[q] = response[:500]

    # ------------------------------------------------------------------
    # 7.  Save report
    # ------------------------------------------------------------------
    print(f"\n[6/6] Saving evaluation report ...")

    report_data = {
        "model_info": model_info,
        "mpt_config": {
            "decay_mode": mpt_cfg.decay_mode,
            "gershgorin_penalty": mpt_cfg.gershgorin_penalty,
            "pseudospectral_epsilon": mpt_cfg.pseudospectral_epsilon,
            "pseudospectral_threshold": mpt_cfg.pseudospectral_threshold,
            "subspace_protection_weight": mpt_cfg.subspace_protection_weight,
            "subspace_rank": mpt_cfg.subspace_rank,
        },
        "data": {
            "source": "https://de.wikipedia.org/wiki/Ansbach",
            "char_len": len(ANSBACH_TEXT),
        },
        "baseline": baseline_report,
        "training": training_report,
        "post_feed": post_report,
        "generations": generations,
    }

    out_path = _SCRIPT_DIR / "evaluation_report_ansbach.json"
    out_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False))
    print(f"   Report saved to {out_path.resolve()}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Model:          {BASE_MODEL}")
    print(f"  MPT decay mode: {mpt_cfg.decay_mode}")
    print(f"  Baseline PPL:   {baseline_ppl:.2f}")
    print(f"  Post-feed PPL:  {post_ppl:.2f}  ({ppl_delta:+.2f} delta)")
    print(f"  Stability gap:  {baseline_stability_gap:.6f} → {post_stability_gap:.6f}")
    print(f"  Gershgorin μ:   {avg_instability:.6f}")
    print(f"  Subspace angle: {subspace_loss.item():.6f}")
    print(f"  Train loss:     {report.loss:.4f}")
    print(f"  Train duration: {report.duration_seconds:.2f}s")
    print(f"  Report saved:   {out_path.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
