"""Pure tensor operations for Matrix Perturbation Theory constructs.

All functions are stateless, fully type-annotated, and operate on raw
tensors — no model references.  Designed for torch.compile compatibility.
"""
from __future__ import annotations

import torch

from rengongtong._types import LoRAWeightMatrix, TensorDict

# ---------------------------------------------------------------------------
# Gershgorin Circle Theorem
# ---------------------------------------------------------------------------


def gershgorin_radii(matrix: LoRAWeightMatrix) -> torch.Tensor:
    """Gershgorin Radius R_i = sum_{j != i} |w_ij| for each row i.

    Returns a zero tensor for non-square matrices (the concept is
    undefined for rectangular matrices).
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return torch.zeros(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    row_sums = torch.sum(torch.abs(matrix), dim=-1)
    diag = torch.abs(torch.diag(matrix))
    return row_sums - diag


def gershgorin_instability(matrix: LoRAWeightMatrix) -> torch.Tensor:
    """Instability = max(0, R_i - |w_ii|).  Non-zero entries mark unstable rows.

    Returns zeros for non-square matrices.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return torch.zeros(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    radii = gershgorin_radii(matrix)
    diag = torch.abs(torch.diag(matrix))
    return torch.clamp(radii - diag, min=0.0)


def _decay_factor(lambda_base: float, penalty: float, instability: torch.Tensor) -> torch.Tensor:
    """Safe decay factor: clamps to [0, 1) so weights never grow."""
    raw = lambda_base + penalty * instability
    return torch.clamp(raw, min=0.0, max=1.0 - 1e-8)


def apply_gershgorin_decay(
    matrix: LoRAWeightMatrix,
    lambda_base: float,
    penalty: float = 0.1,
) -> LoRAWeightMatrix:
    """Scale off-diagonal elements by (1 - decay) proportional to instability.

    Diagonally dominant rows (R_i < |w_ii|) decay at lambda_base rate.
    Rows where the Gershgorin disc extends past the diagonal get an
    extra penalty on their off-diagonal entries.

    Only applies to square matrices (2-D, m == n). Non-square matrices
    decay at uniform lambda_base rate (identity protection only).
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return matrix * (1.0 - lambda_base)

    instability = gershgorin_instability(matrix)
    decay = _decay_factor(lambda_base, penalty, instability)
    return matrix * (1.0 - decay)


# ---------------------------------------------------------------------------
# Gershgorin for LoRA merged weights
# ---------------------------------------------------------------------------


def _match_lora_pairs(weights: TensorDict) -> list[tuple[str, LoRAWeightMatrix, str, LoRAWeightMatrix]]:
    """Group LoRA A/B weight tensors into (name_A, A, name_B, B) pairs.

    Assumes naming pattern: ``...lora_A.default.weight`` / ``...lora_B.default.weight``.
    """
    a_map: dict[str, tuple[str, LoRAWeightMatrix]] = {}
    b_map: dict[str, tuple[str, LoRAWeightMatrix]] = {}
    for key, tensor in weights.items():
        if "lora_A" in key:
            prefix = key.replace("lora_A.default.weight", "")
            a_map[prefix] = (key, tensor)
        elif "lora_B" in key:
            prefix = key.replace("lora_B.default.weight", "")
            b_map[prefix] = (key, tensor)

    pairs: list[tuple[str, LoRAWeightMatrix, str, LoRAWeightMatrix]] = []
    for prefix in a_map:
        if prefix in b_map:
            pairs.append((*a_map[prefix], *b_map[prefix]))
    return pairs


def gershgorin_lora_instability(
    weights: TensorDict,
) -> dict[str, torch.Tensor]:
    """Gershgorin instability of the merged B@A update for each LoRA pair.

    For each matched (A, B) pair, computes ``M = B @ A`` (square for
    attention projections like q/k/v/o where A.shape[1] == B.shape[0])
    and returns the row-wise instability of that merged matrix.
    """
    result: dict[str, torch.Tensor] = {}
    for key_a, A, key_b, B in _match_lora_pairs(weights):
        merged = B @ A
        if merged.shape[0] != merged.shape[1]:
            continue
        result[key_a] = gershgorin_instability(merged)
    return result


def apply_lora_gershgorin_decay(
    weights: TensorDict,
    lambda_base: float,
    penalty: float = 0.1,
) -> TensorDict:
    """Apply Gershgorin-constrained decay to matched LoRA A/B pairs.

    Computes ``M = B @ A``, measures instability of M, then scales
    *both* A and B proportional to M's instability.  Non-paired or
    non-square-merged tensors get uniform decay.
    """
    instabilities = gershgorin_lora_instability(weights)
    result: TensorDict = {}

    for key, tensor in weights.items():
        if key in instabilities:
            inst = instabilities[key]
            decay = _decay_factor(lambda_base, penalty, inst)
            result[key] = tensor * (1.0 - decay)
        else:
            result[key] = tensor * (1.0 - lambda_base)

    return result


# ---------------------------------------------------------------------------
# Pseudospectral Sensitivity
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _inject_lora_noise(model: torch.nn.Module, epsilon: float) -> None:
    """Add ε * N(0,1) in-place to every LoRA parameter."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            noise = torch.randn_like(param) * epsilon
            param.add_(noise)


@torch.inference_mode()
def _remove_lora_noise(model: torch.nn.Module, epsilon: float) -> None:
    """Remove ε * N(0,1) from every LoRA parameter (revert _inject)."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            noise = torch.randn_like(param) * epsilon
            param.sub_(noise)


@torch.inference_mode()
def pseudospectral_sensitivity(
    forward_fn: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    epsilon: float = 1e-5,
    labels: torch.Tensor | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Resolvent-style norm via finite perturbation — self-cleaning.

    The resolvent ‖(zI - W)^{-1}‖ is approximated by:
      1. Forward pass through the model → logits_1
      2. Inject ε-scale noise into every LoRA parameter
      3. Forward pass again → logits_2
      4. Remove noise
      5. Return ‖logits_1 - logits_2‖_F (normalized by ‖logits_1‖_F
         when *normalize* is True, making the metric scale-invariant).

    A large return value signals a *pseudospectral danger zone* where tiny
    weight perturbations cause disproportionately large output changes.
    """
    out1 = forward_fn(input_ids, attention_mask=attention_mask, labels=labels)
    logits_1 = out1.logits if hasattr(out1, "logits") else out1
    norm_1 = torch.norm(logits_1)

    _inject_lora_noise(forward_fn, epsilon)

    out2 = forward_fn(input_ids, attention_mask=attention_mask, labels=labels)
    logits_2 = out2.logits if hasattr(out2, "logits") else out2

    _remove_lora_noise(forward_fn, epsilon)

    gap = torch.norm(logits_1 - logits_2)
    return gap / norm_1 if normalize else gap


# ---------------------------------------------------------------------------
# Davis-Kahan sin-Θ Theorem (Subspace Alignment)
# ---------------------------------------------------------------------------


def subspace_angle(
    V_base: torch.Tensor,
    V_soul: torch.Tensor,
) -> torch.Tensor:
    """Sine of the principal angle between two subspaces (Davis-Kahan sin Θ).

    sin θ = ‖sin Θ(V_base, V_soul)‖ = ‖(I - V_base V_baseᵀ) V_soul‖_F

    Returns the Frobenius-norm sine of the smallest principal angle.
    0  → subspaces are perfectly aligned (no rotation).
    1  → subspaces are fully orthogonal (complete drift).
    """
    projection = V_base @ V_base.T
    orthogonal_component = V_soul - projection @ V_soul
    return torch.norm(orthogonal_component, p="fro")


def subspace_rotation_loss(
    base_weights: TensorDict,
    current_weights: TensorDict,
    rank: int = 8,
) -> torch.Tensor:
    """Davis-Kahan sin-Θ loss penalising rotation of principal subspaces.

    For each LoRA layer in common between *base_weights* and
    *current_weights*, computes the sin of the angle between the
    top-*rank* right singular vectors and returns the mean across
    all layers.
    """
    total = torch.tensor(0.0)
    count = 0

    for key in base_weights:
        if key not in current_weights:
            continue

        W_base = base_weights[key].float()
        W_cur = current_weights[key].float()

        if torch.equal(W_base, W_cur):
            continue

        effective_rank = min(rank, W_base.shape[-1], W_cur.shape[-1])
        if effective_rank < 1:
            continue

        _, _, Vh_base = torch.linalg.svd(W_base, full_matrices=False)
        _, _, Vh_cur = torch.linalg.svd(W_cur, full_matrices=False)

        V_base = Vh_base[:effective_rank].T
        V_cur = Vh_cur[:effective_rank].T

        total = total + subspace_angle(V_base, V_cur)
        count += 1

    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Saliency (shared utility)
# ---------------------------------------------------------------------------


def compute_saliency(weights: TensorDict) -> TensorDict:
    """Saliency = magnitude relative to layer max.

    Weights with high absolute magnitude are deemed 'core' and
    should be protected from decay.
    """
    return {
        name: w.abs() / (w.abs().max() + 1e-8)
        for name, w in weights.items()
    }
