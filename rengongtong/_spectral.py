"""Pure tensor operations for Matrix Perturbation Theory constructs.

All functions are stateless, fully type-annotated, and operate on raw
tensors — no model references.  Designed for torch.compile compatibility.
"""
from __future__ import annotations

import math

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
    """Gershgorin instability of the merged LoRA update for each A/B pair.

    Tries both ``B @ A`` and ``A @ B`` multiplication orders to handle
    different weight storage conventions (Unsloth vs PEFT).  Only square
    merged matrices are scored (attention projections q/k/v/o).
    """
    result: dict[str, torch.Tensor] = {}
    for key_a, A, key_b, B in _match_lora_pairs(weights):
        merged: torch.Tensor | None = None
        try:
            merged = B @ A
        except RuntimeError:
            try:
                merged = A @ B
            except RuntimeError:
                continue
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
    # Build a lookup for B‑key → A‑key (same instability)
    b_to_a: dict[str, str] = {}
    for key_a, A, key_b, B in _match_lora_pairs(weights):
        b_to_a[key_b] = key_a
    result: TensorDict = {}

    for key, tensor in weights.items():
        lookup = key
        if key not in instabilities and key in b_to_a:
            lookup = b_to_a[key]
        if lookup in instabilities:
            inst = instabilities[lookup]
            decay = _decay_factor(lambda_base, penalty, inst)
            # Broadcast decay to match the weight's orientation:
            #   A-weights (r, h) ← decay (h,) → unsqueeze(0) → (1, h)
            #   B-weights (h, r) ← decay (h,) → unsqueeze(-1) → (h, 1)
            if key in instabilities:
                reshaped = (1.0 - decay).unsqueeze(0)
            else:
                reshaped = (1.0 - decay).unsqueeze(-1)
            result[key] = tensor * reshaped
        else:
            result[key] = tensor * (1.0 - lambda_base)

    return result


# ---------------------------------------------------------------------------
# Phase 3 — Mechanism discrimination controls
# ---------------------------------------------------------------------------


def apply_diagonal_mass_decay(
    weights: TensorDict,
    lambda_base: float,
    penalty: float = 0.1,
) -> TensorDict:
    """Uniform off-diagonal penalty on merged BA (no per-row instability).

    Uses the *mean* Gershgorin instability across all rows as a single
    scalar penalty for every row.  If this matches per-row Gershgorin
    decay empirically, the per-row distribution of instability carries
    no additional information — the mechanism is simply
    "off-diagonal → bad, penalize everything equally."
    """
    result: TensorDict = {}
    for key_a, A, key_b, B in _match_lora_pairs(weights):
        merged = B @ A
        if merged.shape[0] != merged.shape[1]:
            result[key_a] = A * (1.0 - lambda_base)
            result[key_b] = B * (1.0 - lambda_base)
            continue

        inst = gershgorin_instability(merged)         # per-row
        mean_inst = float(inst.mean().item())          # scalar: mean instability
        decay_val = min(lambda_base + penalty * mean_inst, 1.0 - 1e-8)

        # Same decay factor for EVERY row
        result[key_a] = A * (1.0 - decay_val)
        result[key_b] = B * (1.0 - decay_val)

    for key, tensor in weights.items():
        if key not in result:
            result[key] = tensor * (1.0 - lambda_base)

    return result


def _random_orthogonal(n: int, device: torch.device | None = None) -> torch.Tensor:
    """Generate a random n×n orthogonal matrix via QR decomposition."""
    H = torch.randn(n, n, device=device)
    Q, R = torch.linalg.qr(H)
    # Ensure Q is orthogonal (det = ±1)
    return Q * torch.sign(torch.diag(R))


def apply_rotated_gershgorin_decay(
    weights: TensorDict,
    lambda_base: float,
    penalty: float = 0.1,
) -> TensorDict:
    """Gershgorin decay applied in a random orthogonal basis.

    For each layer, draws a fixed random orthogonal Q (seeded deterministically
    by layer name so the same rotation is used across ticks) and applies
    Gershgorin decay to Q^T M Q, then maps back.

    Gershgorin discs are basis-dependent.  If the rotated version works
    equally well, the "individual hidden dimensions carry semantic roles"
    story in §5.1 is falsified.
    """
    result: TensorDict = {}

    for key_a, A, key_b, B in _match_lora_pairs(weights):
        merged = B @ A
        d = merged.shape[0]
        if d != merged.shape[1]:
            result[key_a] = A * (1.0 - lambda_base)
            result[key_b] = B * (1.0 - lambda_base)
            continue

        # Deterministic rotation: hash layer name to seed
        seed = abs(hash(key_a)) % (2 ** 31)
        torch.manual_seed(seed)
        Q = _random_orthogonal(d, device=merged.device)

        # Rotate: M_rot = Q^T M Q
        M_rot = Q.T @ merged @ Q

        # Apply Gershgorin decay in rotated basis
        instability = gershgorin_instability(M_rot)
        decay = _decay_factor(lambda_base, penalty, instability)
        M_rot_decayed = M_rot * (1.0 - decay)

        # Rotate back: M' = Q M_rot_decayed Q^T
        M_prime = Q @ M_rot_decayed @ Q.T

        # Project decay back onto A and B
        # We want B' @ A' ≈ M_prime.  Scale both proportionally.
        # Compute a per-row scale factor for M
        with torch.no_grad():
            row_norm_old = torch.norm(merged, dim=-1)
            row_norm_new = torch.norm(M_prime, dim=-1)
            scale = torch.where(
                row_norm_old > 1e-8,
                row_norm_new / row_norm_old,
                torch.ones_like(row_norm_old),
            )
        # Apply scale to B (row-wise) and A (column-wise) proportionally
        scale_sqrt = scale.sqrt()
        result[key_a] = A * scale_sqrt.unsqueeze(0)
        result[key_b] = B * scale_sqrt.unsqueeze(-1)

    for key, tensor in weights.items():
        if key not in result:
            result[key] = tensor * (1.0 - lambda_base)

    return result


# ---------------------------------------------------------------------------
# Pseudospectral Sensitivity
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _inject_lora_noise(model: torch.nn.Module, epsilon: float) -> dict[str, torch.Tensor]:
    """Add ε * N(0,1) in-place to every LoRA parameter. Returns noise dict for removal."""
    noise_dict: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            noise = torch.randn_like(param) * epsilon
            param.add_(noise)
            noise_dict[name] = noise
    return noise_dict


@torch.inference_mode()
def _remove_lora_noise(model: torch.nn.Module, noise_dict: dict[str, torch.Tensor]) -> None:
    """Remove previously injected noise from every LoRA parameter."""
    for name, param in model.named_parameters():
        if name in noise_dict:
            param.sub_(noise_dict[name])


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
    norm_1 = logits_1.float().norm()

    noise_dict = _inject_lora_noise(forward_fn, epsilon)

    out2 = forward_fn(input_ids, attention_mask=attention_mask, labels=labels)
    logits_2 = out2.logits if hasattr(out2, "logits") else out2

    _remove_lora_noise(forward_fn, noise_dict)

    gap = (logits_1.float() - logits_2.float()).norm()
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
    device = None
    for v in base_weights.values():
        device = v.device
        break
    if device is None:
        return torch.tensor(0.0)
    total = torch.tensor(0.0, device=device)
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


# ---------------------------------------------------------------------------
# DCT-based spectral sparsity (SpectralLoRA-style)
# ---------------------------------------------------------------------------


def _dct_matrix(n: int, device: torch.device | None = None) -> torch.Tensor:
    """DCT type-II matrix of size n×n:  T[k, i] = cos(π/n · (i + 0.5) · k)."""
    i = torch.arange(n, device=device)
    k = torch.arange(n, device=device)
    return torch.cos(math.pi / n * (i + 0.5) * k[:, None])


def dct_2d(tensor: LoRAWeightMatrix) -> LoRAWeightMatrix:
    """2D Discrete Cosine Transform (type-II) via separable matrix multiplication.

    D = T_N @ X @ T_M^T   where T_N[k,i] = cos(π/N·(i+½)·k), T_M[l,j] = cos(π/M·(j+½)·l)
    """
    N, M = tensor.shape
    T_n = _dct_matrix(N, tensor.device)
    T_m = _dct_matrix(M, tensor.device)
    return T_n @ tensor.float() @ T_m.T


def idct_2d(coeff: LoRAWeightMatrix) -> LoRAWeightMatrix:
    """2D Inverse Discrete Cosine Transform (type-III).

    X = T_N^T · B_N · D · B_M · T_M
    where B_N = diag(1/N, 2/N, ..., 2/N) and B_M = diag(1/M, 2/M, ..., 2/M).
    """
    N, M = coeff.shape
    T_n = _dct_matrix(N, coeff.device)
    T_m = _dct_matrix(M, coeff.device)
    B_n = torch.diag(torch.cat([
        torch.tensor([1.0 / N], device=coeff.device),
        torch.full([N - 1], 2.0 / N, device=coeff.device),
    ]))
    B_m = torch.diag(torch.cat([
        torch.tensor([1.0 / M], device=coeff.device),
        torch.full([M - 1], 2.0 / M, device=coeff.device),
    ]))
    return T_n.T @ B_n @ coeff.float() @ B_m @ T_m


def spectral_energy_ratio(tensor: LoRAWeightMatrix, k: float = 0.5) -> float:
    """Fraction of total spectral energy captured by the lowest k-fraction of DCT coefficients.

    Low-frequency DCT coefficients (small indices) capture smooth structure.
    A high energy ratio for small k means the weight is spectrally sparse.
    """
    if k <= 0 or k > 1:
        return 1.0
    coeff = dct_2d(tensor)
    flat = coeff.abs().reshape(-1)
    n_keep = max(1, int(k * flat.numel()))
    top = flat.topk(n_keep, largest=True)
    return (top.values.sum() / flat.sum()).item()


def compress_lora_weight(tensor: LoRAWeightMatrix, retention: float = 0.5) -> LoRAWeightMatrix:
    """Zero out high-frequency DCT coefficients, keeping only the lowest *retention* fraction.

    *retention* = 1.0 → no compression (identity).
    *retention* = 0.5 → keep 50% lowest-frequency coefficients.
    """
    if retention >= 1.0:
        return tensor.clone()
    coeff = dct_2d(tensor)
    flat = coeff.abs().reshape(-1)
    n_keep = max(1, int(retention * flat.numel()))
    threshold = flat.kthvalue(n_keep + 1).values
    mask = coeff.abs() >= threshold
    coeff[~mask] = 0.0
    return idct_2d(coeff).to(tensor.dtype)


def lora_spectral_summary(weights: TensorDict) -> dict[str, float]:
    """Aggregate DCT spectral sparsity metrics across all LoRA weights."""
    energy_ratios = {}
    for key, tensor in weights.items():
        if "lora_" in key and tensor.ndim == 2:
            for retention in [0.1, 0.25, 0.5]:
                tag = f"{key}/energy_r{retention:.0%}"
                t = tensor
                if t.numel() > 256 * 256:
                    t = t[:256, :256]
                energy_ratios[tag] = round(spectral_energy_ratio(t, retention), 4)
    return energy_ratios


def spectral_projection_step(
    model: torch.nn.Module,
    retention: float,
    skip_bias: bool = True,
) -> dict[str, float]:
    """After-optimizer projection: DCT-compress all LoRA A/B matrices in-place.

    Returns per-weight compression ratios (fraction of coeffs zeroed).
    """
    metrics = {}
    for name, param in model.named_parameters():
        if "lora_" not in name or param.ndim != 2:
            continue
        if skip_bias and param.ndim < 2:
            continue
        compressed = compress_lora_weight(param.data, retention)
        zeroed = (compressed == 0).count_nonzero().item()
        total = param.numel()
        param.data.copy_(compressed)
        metrics[name] = round(zeroed / total, 4)
    return metrics


# ---------------------------------------------------------------------------
# Adaptive rank (AdaLoRA-style pruning during training)
# ---------------------------------------------------------------------------


def effective_rank(
    B: LoRAWeightMatrix,
    A: LoRAWeightMatrix,
    variance_threshold: float = 0.95,
) -> int:
    """Minimum number of singular values needed to retain *variance_threshold* of total variance.

    Uses low-rank QR trick: ΔW = B @ A, computes SVD of r×r matrix.
    """
    if B.ndim != 2 or A.ndim != 2:
        return max(B.shape[0] if B.ndim == 2 else 1, A.shape[0] if A.ndim == 2 else 1)
    try:
        _, Rb = torch.linalg.qr(B.float(), mode="reduced")
        _, Ra = torch.linalg.qr(A.T.float(), mode="reduced")
        S = torch.linalg.svdvals(Rb @ Ra.T)
        total = S.sum() + 1e-10
        if total < 1e-8:
            return 1
        cum = S.cumsum(0)
        idx = (cum / total >= variance_threshold).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return B.shape[1]
        return max(1, int(idx[0].item()) + 1)
    except RuntimeError:
        return B.shape[1]


def effective_rank_summary(
    model: torch.nn.Module,
    variance_threshold: float = 0.95,
) -> dict[str, int]:
    """Aggregate effective rank per-layer across all LoRA weight pairs."""
    b_tensors: dict[str, torch.Tensor] = {}
    a_tensors: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        # Strip adapter suffix and param leaf, e.g. "lora_B.default.weight" → "lora_B"
        clean = name.replace(".default", "")
        kind = clean.split(".")[-2]  # lora_A or lora_B
        parent = clean.rsplit(".", 2)[0]
        if kind == "lora_B":
            b_tensors[parent] = param.data
        elif kind == "lora_A":
            a_tensors[parent] = param.data

    ranks = {}
    for parent in b_tensors:
        if parent in a_tensors:
            r = effective_rank(b_tensors[parent], a_tensors[parent], variance_threshold)
            ranks[parent] = r
    return ranks


def prune_lora_rank(
    model: torch.nn.Module,
    target_variance: float = 0.95,
) -> dict[str, int]:
    """For each LoRA pair (B, A), replace B and A with truncated SVD.

    This zeroes out the least important singular directions, effectively
    reducing the model's capacity in low-saliency layers.

    Uses the low-rank structure: ΔW = B @ A where B is d×r, A is r×k.
    SVD is computed via QR decomposition for O(d·r² + r³) complexity.

    Returns dict of layer_name → new_effective_rank.
    """
    b_params: dict[str, torch.nn.Parameter] = {}
    a_params: dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        clean = name.replace(".default", "")
        kind = clean.split(".")[-2]
        parent = clean.rsplit(".", 2)[0]
        if kind == "lora_B":
            b_params[parent] = param
        elif kind == "lora_A":
            a_params[parent] = param

    new_ranks = {}
    for parent in b_params:
        if parent not in a_params:
            continue
        B = b_params[parent].data  # d × r
        A = a_params[parent].data  # r × k
        d, r = B.shape
        r2, k = A.shape
        if r != r2:
            continue

        # Efficient SVD of low-rank ΔW = B @ A using QR trick
        try:
            Qb, Rb = torch.linalg.qr(B.float(), mode="reduced")    # d×r, r×r
            Qa, Ra = torch.linalg.qr(A.T.float(), mode="reduced")  # k×r, r×r
            M = Rb @ Ra.T  # r × r
            Um, S, Vtm = torch.linalg.svd(M, full_matrices=False)
            U = Qb @ Um      # d × r
            Vt = Vtm @ Qa.T  # r × k
        except RuntimeError:
            continue

        total = S.sum() + 1e-10
        if total < 1e-8:
            new_ranks[parent] = 1
            continue
        cum = S.cumsum(0)
        idx = (cum / total >= target_variance).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            new_ranks[parent] = r
            continue
        n = max(1, min(int(idx[0].item()) + 1, r - 1))

        if n >= r:
            new_ranks[parent] = r
            continue

        sqrt_S = S[:n].sqrt()
        B[:, :n].copy_(U[:, :n] * sqrt_S[None, :])
        B[:, n:].zero_()
        A[:n, :].copy_(Vt[:n, :] * sqrt_S[:, None])
        A[n:, :].zero_()

        new_ranks[parent] = n

    return new_ranks
