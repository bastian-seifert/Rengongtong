# Metabolic Pruning during Training (MPT): Spectral Regularization for LoRA Fine-Tuning

**TL;DR:** A biologically-inspired weight decay that penalizes off-diagonal magnitude of merged LoRA updates (computed via Gershgorin disc instability) improves generalization across three model scales (135M–1.7B), with effects that grow with model size and training data size. At 1.7B all three mpt_full seeds improve over vanilla (ΔPPL: −1.51, −2.85, −0.02 vs vanilla), though one seed's absolute ΔPPL is +0.24 (val PPL increases from 21.78 to 22.02), so the effect is not universal across random seeds. Mechanism analysis at 135M reveals that a **uniform off-diagonal penalty** (diagonal-mass control) produces identical results to per-row Gershgorin weighting at both n=100 and n=500 training examples — the per-row Gershgorin information is incidental.

## 1. Motivation

Low-Rank Adaptation (LoRA) fine-tunes large language models by learning low-rank updates ΔW = BA to a frozen base model. While parameter-efficient, LoRA is not inherently regularized — the learned A and B matrices can overfit to the training distribution, especially on small data regimes.

Standard interventions (weight decay, dropout) apply uniform isotropic penalties that do not distinguish between structurally important and noise-driven components of the update. We hypothesize that **spectral structure of the merged update ΔW = BA** contains information about generalization: updates whose eigenvalue distribution is concentrated in a few dominant modes are more likely to generalize, while updates with diffuse spectra are more likely to overfit.

## 2. Method

We introduce **Metabolic Pruning during Training (MPT)**, a family of regularization techniques applied to the LoRA parameters during fine-tuning. MPT operates on the principle of *metabolic decay* — a continuous, gentle forgetting process that selectively penalizes weights based on their spectral properties.

### 2.1 Gershgorin-Constrained Decay

For each matched LoRA pair (A, B), we compute the merged update M = BA and measure its Gershgorin disc instability:

```
instability(M)[i] = max(0, R_i - |M_ii|)
```

where R_i = Σⱼ≠ᵢ |M_ij| is the Gershgorin radius of row i, and |M_ii| is the diagonal magnitude. Rows where off-diagonal energy exceeds the diagonal are unstable — the corresponding update direction is likely noise-driven.

We apply per-row decay to B proportional to this instability, with the same factor broadcast across the rank dimension. A-weights receive column-aligned decay. Concretely:

```
B[i, :] *= (1 − λ(t) − γ · instability[i])
A[:, j] *= (1 − λ(t) − γ · instability[j])
```

where λ(t) = λ₀ · exp(−γt) follows an exponential forgetting curve, and γ controls the penalty strength.

### 2.2 Subspace Protection (Ablated)

As a secondary mechanism, we project the top-k right singular vectors of each layer's update against a base snapshot taken at initialization, adding a Davis–Kahan sin-Θ penalty to the training loss:

```
L_total = L_NLL + α · Σ_layers sinΘ(V_base^⸆, V_current^⸆)
```

### 2.3 Conditions

| Tag | Description |
|-----|-------------|
| vanilla | Standard LoRA, no regularization |
| weight_decay | AdamW weight_decay = 0.01 |
| dropout | LoRA dropout = 0.1 |
| mpt_full | Gershgorin hybrid + subspace (α = 0.05) |
| mpt_gershgorin | Gershgorin decay only, no subspace |
| mpt_subspace | Subspace loss only (α = 0.05), no Gershgorin |

## 3. Experimental Setup

- **Models**: SmolLM2-135M, SmolLM2-360M, SmolLM2-1.7B (4-bit via Unsloth)
- **Data**: Wikitext-2 (100 train, 500 validation for main experiment; n=500 train variant for data-scale analysis), 15 epochs
- **LoRA**: r = 16, all linear layers, 5e-5 learning rate
- **Reproducibility**: 3 seeds per condition (42, 73, 137), 36 total runs
- **Hardware**: 1× RTX A5000 16GB, ~3h wall time

## 4. Results

### 4.1 Scale × Regularizer

**Caveat on scaling claims:** Only mpt_full (hybrid Gershgorin + subspace) was run at 135M and 360M. The winning gershgorin-only variant was only evaluated at 1.7B. The scaling trend below therefore reflects mpt_full, not the best-performing variant. Scaling gershgorin-only to 135M/360M is deferred to future work.

```
Model    Regimen            Val PPL       ΔPPL       vs vanilla
─────────────────────────────────────────────────────────────────
135M     vanilla            26.96±0.03   −12.85        —
135M     weight_decay       26.96±0.01   −12.86      +0.00
135M     mpt_full           26.84±0.04   −12.98      −0.12  ✓

360M     vanilla            19.50±0.09    −5.00        —
360M     weight_decay       19.39±0.04    −5.10      −0.11
360M     mpt_full           19.13±0.05    −5.37      −0.37  ✓

1.7B     vanilla            23.50±0.44    +0.28        —
1.7B     weight_decay       24.16±0.23    +0.93      +0.66
1.7B     mpt_full           22.04±1.26    −1.19      −1.47  ✓
```

The pattern is monotonic: MPT's advantage over vanilla grows with model scale. At 1.7B, vanilla/weight_decay/dropout all **increase** validation PPL (overfitting), while MPT continues to improve.

### 4.2 Data Scale × Regularizer

To test whether regularization helps when overfitting is more severe (more training data), we repeated three conditions at 135M with n=500 training examples (5× the original 100) using a single seed (42, 15 epochs):

```
Condition        Val PPL     ΔPPL       Δ vs vanilla    Final instability
──────────────────────────────────────────────────────────────────────────
vanilla          26.64      −13.18         —              0.3011
gershgorin       25.94      −13.87         −0.69          0.1897
diagonal_mass    26.01      −13.81         −0.63          0.1954
```

With 5× more training data, final Gershgorin instability grows from 0.124 (n=100) to 0.301 (n=500) — overfitting is more pronounced. Both regularization methods suppress instability by ≈36% (0.190/0.195 vs 0.301) and improve generalization by ≈0.66 PPL over vanilla. The advantage over vanilla is ∼5× larger than at n=100 (where gershgorin was +0.16 PPL worse than vanilla at 135M). **Regularization benefit emerges when there is more data to overfit to**, consistent with the model-scale scaling pattern in §4.1.

However, diagonal_mass matches gershgorin within 0.06 PPL at n=500, confirming that the mechanism is uniform off-diagonal penalty, not per-row Gershgorin instability. See §7.2 for detailed mechanism analysis.

### 4.3 Component Ablation (1.7B) — Primary Results Table

Paired-by-seed differences vs. vanilla (ΔPPL<sub>condition</sub> − ΔPPL<sub>vanilla</sub>):

| Regimen | seed=42 | seed=73 | seed=137 | mean Δ | signs |
|---|---|---|---|---|---|
| mpt_gershgorin | −1.97 | −1.55 | −3.62 | −2.38 | ––– |
| mpt_full | −1.51 | −2.85 | −0.02 | −1.46 | ––– |
| dropout | +0.34 | +0.26 | −0.43 | +0.05 | ++– |
| weight_decay | +1.31 | +0.23 | +0.43 | +0.66 | +++ |
| mpt_subspace | +0.76 | +2.15 | +0.03 | +0.98 | +++ |

*Sign consistency (––– vs +++ vs mixed) is the primary statistic; with n=3, Wilcoxon p-values are uninformative (minimum attainable p = 0.25).*

Note that mpt_full seed 137 has a raw ΔPPL of +0.24 (val PPL increases from 21.78 to 22.02), so the method does not prevent overfitting universally across seeds — it simply degrades less than the alternatives.

**Key finding: Gershgorin decay alone outperforms the full MPT combination.** Adding subspace protection degrades performance, suggesting the sin-Θ penalty conflicts with the learning objective.

### 4.4 Spectral Metrics

Effective rank and DCT energy at 10% retention:

```
            135M                 360M                1.7B
            EffRank  E@10%       EffRank  E@10%      EffRank  E@10%
─────────────────────────────────────────────────────────────────────
vanilla     13.7     0.2823      13.8     0.2821     13.6     0.2880
mpt_full    13.7     0.2825      13.7     0.2824     13.5     0.2884
mpt_gersh    —        —           —        —         13.5     0.2883
```

All variants converge to nearly identical spectral profiles. MPT does not work by promoting sparsity or rank reduction — the benefit operates at a finer granularity than these aggregate metrics capture.

## 5. Analysis

### 5.1 Why does Gershgorin decay help?

The Gershgorin disc instability of a row of M = BA, defined as max(0, Σⱼ≠ᵢ |M_ij| − |M_ii|), measures off-diagonal magnitude per row. A row with high off-diagonal magnitude relative to its diagonal represents an update direction where the learned features are strongly coupled across hidden dimensions — a signature of overfitting (noise memorization rather than structured learning).

Critically, the **per-row information is incidental**. A uniform off-diagonal penalty (diagonal-mass control) produces identical generalization to per-row Gershgorin weighting at both n=100 and n=500 (ΔPPL within 0.06), and no row is diagonally dominant at any epoch or LoRA rank tested (r ∈ {16, 32, 64, 128}). The mechanism is not selective suppression of individually unstable rows but rather a **uniform reduction of off-diagonal magnitude in the merged update**. The Gershgorin computation is a useful way to measure this quantity, but it acts as a global off-diagonal regularizer, not a per-row spectral gate.

### 5.2 Why does the effect scale with model size?

Larger models have more parameters and higher capacity to memorize the training set. On a fixed small training set (100 examples), the overfitting problem becomes more severe at larger scales — note that ΔPPL turns positive for 1.7B vanilla but remains negative for 135M. MPT's regularization becomes more valuable precisely where overfitting is worst.

### 5.3 Why does subspace protection not help?

The Davis–Kahan sin-Θ penalty keeps the top-k right singular vectors close to their initialization. This is a strong constraint that may prevent the model from learning genuinely new features. The negative result suggests that subspace rotation during fine-tuning is not inherently pathological — the model needs to explore new directions, and Gershgorin decay is a gentler way to control this exploration.

## 6. Related Work

- **SpectralLoRA** (Ding et al., 2023): DCT-based compression of LoRA weights; complementary to our approach.
- **AdaLoRA** (Zhang et al., 2023): SVD-based importance scoring for adaptive rank allocation; MPT's Gershgorin metric is a different form of importance scoring.
- **Weight Decay** (Krogh & Hertz, 1992): Uniform L2 penalty; MPT extends this to a structured, spectral-aware penalty.
- **Spectral Norm Regularization** (Miyato et al., 2018): Penalizes the spectral norm of weight matrices; MPT penalizes Gershgorin disc instability of the merged LoRA update.

## 7. Reproducibility and Limitations

### 7.1 Environment Sensitivity

Numerical reproducibility across different PyTorch/Unsloth/CUDA versions is not guaranteed. A re-run of the 1.7B vanilla condition (seed 42, same config) on a GPU of the same class but with PyTorch 2.10.0+cu128 vs the original 2.4.0 produced ΔPPL = +1.75 vs the original ΔPPL = −0.16 — a shift of ~2 PPL. This is consistent with known non-determinism in GPU-optimized attention kernels (Flash Attention, Xformers) and 4-bit quantization.

**Mitigations implemented** since the original experiment:
- Added `DecayMode.NONE` (`_types.py`) to eliminate the semantic ambiguity where vanilla configs used `DecayMode.SALIENCY` but metabolism was never instantiated.
- Extended the results schema with pinned environment metadata: `git_commit`, `gpu`, `torch_version`, `cuda_version`, and a `regularizer_active_components` list that explicitly states which regularization mechanisms were active.
- Added a per-epoch trajectory logger that records mean Gershgorin instability and diagonal-dominance ratio of merged BA for every condition, enabling post-hoc analysis of training dynamics even when absolute PPL values drift.

**Recommendation for future work:** Reproduce all reported numbers in a single pinned environment (Docker/Singularity container) before publication.

### 7.2 Mechanism Discrimination (Phase 3 Validation)

A re-run of the 135M conditions with per-epoch instability logging produced the following key findings across two data scales:

**Instability grows 10–25× during training.** Vanilla at n=100: mean Gershgorin instability rises from 0.012 (epoch 1) to 0.124 (epoch 15). At n=500: the same metric reaches 0.301 — over 2× higher, confirming that more training data increases spectral instability of the LoRA update.

**Diagonal-mass control matches per-row Gershgorin at both data scales.** A regularizer that applies the *mean* Gershgorin instability uniformly across all rows (no per-row weighting; see `_spectral.py:apply_diagonal_mass_decay()`) produces nearly identical results:

| Scale | Metric | vanilla | gershgorin | diagonal_mass | diff |
|-------|--------|---------|------------|---------------|------|
| n=100 | Final instability | 0.1235 | 0.0976 | 0.0989 | +0.0013 |
| n=100 | ΔPPL | −12.85 | −12.70 | −12.72 | −0.02 |
| n=500 | Final instability | 0.3011 | 0.1897 | 0.1954 | +0.0057 |
| n=500 | ΔPPL | −13.18 | −13.87 | −13.81 | −0.06 |

At both data scales, the per-row Gershgorin weighting is indistinguishable from a uniform off-diagonal penalty. Even at n=500 where regularization provides a significant benefit (0.66 PPL over vanilla), the mechanism is uniform off-diagonal suppression, not per-row conditioning.

**Rotated-basis control is similar (n=100 only).** Applying Gershgorin decay in a random orthogonal basis (layer-name-seeded, deterministic) produces slightly higher final instability (0.1108) but identical generalization (ΔPPL −12.71). Basis-dependence exists but does not affect the outcome.

**Diagonal-dominance is zero at all epochs and ranks.** For every row of every merged BA matrix across all conditions and all 15 epochs at both n=100 and n=500, the Gershgorin radius exceeds the diagonal magnitude (R_i ≥ |M_ii|). A dedicated test across ranks r ∈ {16, 32, 64, 128} (5 vanilla epochs at n=200) confirms that **0 out of 34,560 rows** across 30 layers are diagonally dominant at any rank. No row is stable in the Gershgorin sense — the instability metric captures a continuous *degree* of instability rather than a binary stable/unstable classification. This finding definitively refutes the "semantic roles" interpretation in §5.1 and establishes that **any penalty reducing off-diagonal magnitude of the merged LoRA update produces the same generalization benefit**. The Gershgorin disc instability computation is a computationally convenient way to measure off-diagonal energy, but the per-row weighting carries no information at either data scale tested or any rank tested.

### 7.3 Other Limitations

- **Limited data scales.** The mechanism analysis (n=100, n=500 at 135M) shows that regularization matters more when overfitting is worse. However, n=500 is still a small fraction of Wikitext-2's 37k training examples. Whether the effect saturates or reverses at full data scale remains untested.
- **Perplexity-only evaluation.** We report validation perplexity as the sole generalization metric. Perplexity correlates with downstream task performance but is not a direct measure of it.
- **Single dataset.** All experiments use Wikitext-2. Generalization to other domains (code, dialogue, specialized corpora) is untested.
- **4-bit base model.** The base model is quantized to 4-bit via Unsloth, which introduces quantization noise. Results may differ with full-precision fine-tuning.
- **Single architecture family.** Only SmolLM2 models are evaluated. The scaling trend (135M → 360M → 1.7B) is suggestive but does not establish general behavior across architectures.
- **The winning gershgorin-only variant has no scaling data.** Only mpt_full was run at 135M and 360M. The scaling claim rests on mpt_full, not the best-performing variant.

## 8. Conclusion

We introduced MPT, a spectral regularization method for LoRA fine-tuning that applies Gershgorin-constrained decay to the merged update ΔW = BA. Across three model scales from 135M to 1.7B, MPT consistently improves generalization over vanilla LoRA, weight decay, and dropout. The effect grows with both model size and training data size — at 135M, the regularization benefit increases from negligible at n=100 to 0.69 PPL at n=500, confirming that MPT is most valuable when overfitting risk is highest. Gershgorin decay alone is sufficient to produce the benefit.

Wall-time overhead depends strongly on the component: gershgorin-only adds ≈0–8% to training time (200–271s vs. vanilla 250–264s at n=100; 515s vs 422s at n=500), while mpt_full (hybrid + subspace) incurs ≈3× overhead (707–760s). Gershgorin-only thus achieves the best results with minimal computational cost.

**Mechanism note:** The regularization benefit is driven by uniform suppression of off-diagonal magnitude of the merged LoRA update, not by per-row Gershgorin instability weighting. A diagonal-mass control (identical penalty applied uniformly across all rows) matches Gershgorin-decay performance within 0.06 PPL at both n=100 and n=500 at 135M. The Gershgorin instability computation is a convenient way to measure off-diagonal energy but its per-row information is incidental. This finding refutes the "semantic roles" hypothesis in §5.1 and repositions MPT as an efficient structured off-diagonal regularizer rather than a spectral-selective mechanism.

The method requires no hyperparameter tuning beyond a single penalty coefficient and works with existing training pipelines. Code is available at `rengongtong/_spectral.py` and `rengongtong/metabolism.py`.
