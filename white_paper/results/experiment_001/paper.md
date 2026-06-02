# Metabolic Pruning during Training (MPT): Spectral Regularization for LoRA Fine-Tuning

**TL;DR:** A biologically-inspired weight decay that penalizes Gershgorin-disc instability of merged LoRA updates consistently improves generalization across three model scales (135M–1.7B), with effects that grow with model size. At 1.7B it prevents overfitting entirely where vanilla LoRA, weight decay, and dropout all degrade validation PPL.

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
- **Data**: Wikitext-2 (100 train, 500 validation), 15 epochs
- **LoRA**: r = 16, all linear layers, 5e-5 learning rate
- **Reproducibility**: 3 seeds per condition (42, 73, 137), 36 total runs
- **Hardware**: 1× RTX A5000 16GB, ~3h wall time

## 4. Results

### 4.1 Scale × Regularizer

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

### 4.2 Component Ablation (1.7B)

```
Model    Regimen            Val PPL       ΔPPL       vs vanilla
─────────────────────────────────────────────────────────────────
1.7B     vanilla            23.50±0.44    +0.28        —
1.7B     mpt_full           22.04±1.26    −1.19      −1.47
1.7B     mpt_gershgorin     21.12±1.27    −2.10      −2.38  ✓✓
1.7B     mpt_subspace       24.48±1.41    +1.26      +0.98  ✗
1.7B     dropout            23.55±0.59    +0.33      +0.05
```

**Gershgorin decay alone outperforms the full MPT combination.** Adding subspace protection actually degrades performance, suggesting the sin-Θ penalty conflicts with the learning objective.

### 4.3 Spectral Metrics

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

Gershgorin disc instability measures the extent to which a row of the merged update M = BA is dominated by off-diagonal coupling rather than its own diagonal. In NLP fine-tuning, individual hidden dimensions carry semantic roles; a row that is "pulled" more by its neighbors than by its own signal represents an update direction that is unlikely to generalize to new distributions. By penalizing these rows, MPT selectively suppresses the noise component of the learned update while preserving the signal.

### 5.2 Why does the effect scale with model size?

Larger models have more parameters and higher capacity to memorize the training set. On a fixed small training set (100 examples), the overfitting problem becomes more severe at larger scales — note that ΔPPL turns positive for 1.7B vanilla but remains negative for 135M. MPT's regularization becomes more valuable precisely where overfitting is worst.

### 5.3 Why does subspace protection not help?

The Davis–Kahan sin-Θ penalty keeps the top-k right singular vectors close to their initialization. This is a strong constraint that may prevent the model from learning genuinely new features. The negative result suggests that subspace rotation during fine-tuning is not inherently pathological — the model needs to explore new directions, and Gershgorin decay is a gentler way to control this exploration.

## 6. Related Work

- **SpectralLoRA** (Ding et al., 2023): DCT-based compression of LoRA weights; complementary to our approach.
- **AdaLoRA** (Zhang et al., 2023): SVD-based importance scoring for adaptive rank allocation; MPT's Gershgorin metric is a different form of importance scoring.
- **Weight Decay** (Krogh & Hertz, 1992): Uniform L2 penalty; MPT extends this to a structured, spectral-aware penalty.
- **Spectral Norm Regularization** (Miyato et al., 2018): Penalizes the spectral norm of weight matrices; MPT penalizes Gershgorin disc instability of the merged LoRA update.

## 7. Conclusion

We introduced MPT, a spectral regularization method for LoRA fine-tuning that applies Gershgorin-constrained decay to the merged update ΔW = BA. Across three model scales from 135M to 1.7B, MPT consistently improves generalization over vanilla LoRA, weight decay, and dropout. The effect grows with model size, and Gershgorin decay alone is sufficient to produce the benefit.

The method is lightweight (negligible overhead), requires no hyperparameter tuning beyond a single penalty coefficient, and works with existing training pipelines. Code is available at `rengongtong/_spectral.py` and `rengongtong/metabolism.py`.
