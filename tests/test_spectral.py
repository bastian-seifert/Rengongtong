"""Tests for _spectral.py — pure MPT tensor math functions."""
from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import pytest
import torch

from rengongtong._spectral import (
    apply_gershgorin_decay,
    apply_lora_gershgorin_decay,
    compute_saliency,
    gershgorin_instability,
    gershgorin_lora_instability,
    gershgorin_radii,
    pseudospectral_sensitivity,
    subspace_angle,
    subspace_rotation_loss,
)


class TestGershgorinRadii:
    def test_radii_diagonally_dominant(self):
        """A diagonally dominant matrix should have zero radii."""
        m = torch.tensor([[5.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 5.0]])
        radii = gershgorin_radii(m)
        expected = torch.tensor([2.0, 2.0, 2.0])  # sum_{j≠i} |w_ij|
        assert torch.allclose(radii, expected)

    def test_radii_identity(self):
        m = torch.eye(3)
        radii = gershgorin_radii(m)
        assert torch.allclose(radii, torch.zeros(3))

    def test_radii_single_element(self):
        m = torch.tensor([[42.0]])
        radii = gershgorin_radii(m)
        assert radii.item() == 0.0

    def test_radii_off_diagonal_dominated(self):
        """Non-zero off-diagonals with zero diagonal."""
        m = torch.tensor([[0.0, 5.0], [3.0, 0.0]])
        radii = gershgorin_radii(m)
        assert torch.allclose(radii, torch.tensor([5.0, 3.0]))

    def test_radii_non_square_returns_zeros(self):
        m = torch.randn(3, 5)
        radii = gershgorin_radii(m)
        assert radii.shape == (3,)
        assert torch.allclose(radii, torch.zeros(3))


class TestGershgorinInstability:
    def test_stable_when_diagonally_dominant(self):
        """R_i < |w_ii| → instability = 0."""
        m = torch.tensor([[10.0, 2.0], [2.0, 10.0]])
        inst = gershgorin_instability(m)
        assert torch.allclose(inst, torch.zeros(2))

    def test_unstable_when_not_dominant(self):
        """R_i > |w_ii| → instability > 0."""
        m = torch.tensor([[1.0, 5.0], [5.0, 1.0]])
        inst = gershgorin_instability(m)
        # Row 0: R=5, |diag|=1 → max(0, 5-1) = 4
        # Row 1: R=5, |diag|=1 → max(0, 5-1) = 4
        assert torch.allclose(inst, torch.tensor([4.0, 4.0]))

    def test_negative_instability_clamped(self):
        m = torch.tensor([[100.0, 1.0], [1.0, 100.0]])
        inst = gershgorin_instability(m)
        assert torch.all(inst >= 0.0)

    def test_instability_non_square_returns_zeros(self):
        m = torch.randn(3, 5)
        inst = gershgorin_instability(m)
        assert inst.shape == (3,)
        assert torch.allclose(inst, torch.zeros(3))


class TestApplyGershgorinDecay:
    def test_reduces_magnitude(self):
        m = torch.tensor([[10.0, 2.0], [2.0, 10.0]])
        result = apply_gershgorin_decay(m, lambda_base=0.1, penalty=0.5)
        assert torch.all(result.abs() < m.abs())

    def test_stable_rows_decay_less(self):
        """Diagonally dominant rows should shrink less than unstable ones."""
        # Row 0: [100, 1] → R=1, |diag|=100, instability=0 (stable)
        # Row 1: [5, 1]   → R=5, |diag|=1, instability=4 (unstable)
        m = torch.tensor([[100.0, 1.0], [5.0, 1.0]])
        result = apply_gershgorin_decay(m, lambda_base=0.01, penalty=0.1)
        assert result[1, 0] < m[1, 0]  # unstable row decays more

    def test_zeros_in_zeros_out(self):
        m = torch.zeros(3, 3)
        result = apply_gershgorin_decay(m, lambda_base=0.1)
        assert torch.allclose(result, torch.zeros(3, 3))

    def test_decay_preserves_dtype(self):
        m = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        result = apply_gershgorin_decay(m, lambda_base=0.1)
        assert result.dtype == torch.float64

    def test_non_square_uses_uniform_decay(self):
        m = torch.randn(3, 5)
        result = apply_gershgorin_decay(m, lambda_base=0.1, penalty=1.0)
        expected = m * 0.9
        assert torch.allclose(result, expected)


class TestSubspaceAngle:
    def test_identical_subspaces_zero_angle(self):
        V = torch.eye(3)[:, :2]
        angle = subspace_angle(V, V)
        assert torch.isclose(angle, torch.tensor(0.0), atol=1e-6)

    def test_orthogonal_subspaces_max_angle(self):
        V_base = torch.eye(3)[:, :2]
        V_soul = torch.eye(3)[:, 2:3]  # last column, orthogonal to first 2
        angle = subspace_angle(V_base, V_soul)
        assert angle > 0.5  # Frobenius norm of orthogonal component

    def test_single_vector(self):
        v = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).T
        angle = subspace_angle(v, v)
        assert torch.isclose(angle, torch.tensor(0.0), atol=1e-6)

    def test_zero_input_does_not_crash(self):
        V_base = torch.zeros(3, 2)
        V_soul = torch.eye(3)[:, :2]
        # Should not raise — undefined but stable
        result = subspace_angle(V_base, V_soul)
        assert torch.isfinite(result)


class TestSubspaceRotationLoss:
    def make_weights(self) -> dict[str, torch.Tensor]:
        return {
            "base.model.lora_A.weight": torch.randn(4, 4),
            "base.model.lora_B.weight": torch.randn(4, 4),
        }

    def test_same_weights_zero_loss(self):
        w = self.make_weights()
        loss = subspace_rotation_loss(w, w, rank=2)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)

    def test_different_weights_positive_loss(self):
        base = self.make_weights()
        current = {k: v + torch.randn_like(v) * 0.5 for k, v in base.items()}
        loss = subspace_rotation_loss(base, current, rank=2)
        assert loss > 0.0

    def test_partial_overlap(self):
        base = {"a": torch.randn(3, 3)}
        current = {"b": torch.randn(3, 3)}  # different key
        loss = subspace_rotation_loss(base, current, rank=2)
        assert loss.item() == 0.0  # no common keys

    def test_rank_clamped_to_matrix_dim(self):
        base = {"a": torch.randn(2, 2)}
        current = {"a": base["a"] + torch.randn_like(base["a"]) * 0.1}
        loss = subspace_rotation_loss(base, current, rank=100)  # rank > dim
        assert torch.isfinite(loss)


class TestPseudospectralSensitivity:
    def test_identical_forward_returns_zero(self):
        model = MagicMock()
        model.named_parameters = MagicMock(return_value=[])
        model.logits = torch.randn(1, 5, 10)

        def fake_forward(input_ids, **kw):
            return MagicMock(logits=model.logits)

        model.side_effect = fake_forward
        model.__call__ = MagicMock(side_effect=fake_forward)

        input_ids = torch.randint(0, 100, (1, 5))
        gap = pseudospectral_sensitivity(model, input_ids, epsilon=0.0)
        assert gap.item() == 0.0

    def test_no_lora_params_does_not_crash(self):
        model = MagicMock()
        model.named_parameters = MagicMock(return_value=[])
        model.logits = torch.randn(1, 5, 10)

        def fake_forward(input_ids, **kw):
            return MagicMock(logits=model.logits)

        model.side_effect = fake_forward
        model.__call__ = MagicMock(side_effect=fake_forward)

        input_ids = torch.randint(0, 100, (1, 5))
        gap = pseudospectral_sensitivity(model, input_ids, epsilon=1e-5)
        assert gap.item() == 0.0


class TestComputeSaliency:
    def test_max_weight_is_one(self):
        w = {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        sal = compute_saliency(w)
        assert torch.max(sal["a"]).item() == pytest.approx(1.0)

    def test_min_weight_non_negative(self):
        w = {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        sal = compute_saliency(w)
        assert torch.min(sal["a"]).item() >= 0.0

    def test_zero_weights_dont_crash(self):
        """Should handle all-zero weights without division by zero."""
        w = {"a": torch.zeros(3, 4)}
        sal = compute_saliency(w)
        assert torch.allclose(sal["a"], torch.zeros(3, 4))


class TestLoRAGershgorin:
    def make_lora_weights(self) -> dict[str, torch.Tensor]:
        """Simulates a typical LoRA pair: A=(r×dim), B=(dim×r)."""
        return {
            "base.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(16, 64),
            "base.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(64, 16),
        }

    def test_match_lora_pairs(self):
        from rengongtong._spectral import _match_lora_pairs
        w = self.make_lora_weights()
        pairs = _match_lora_pairs(w)
        assert len(pairs) == 1
        key_a, A, key_b, B = pairs[0]
        assert "lora_A" in key_a
        assert "lora_B" in key_b

    def test_gershgorin_lora_instability_merged_square(self):
        """B@A is 64×64 square, so instability should be computed."""
        w = self.make_lora_weights()
        inst = gershgorin_lora_instability(w)
        assert len(inst) == 1
        key = next(iter(inst))
        assert inst[key].shape == (64,)

    def test_apply_lora_gershgorin_decay_preserves_keys(self):
        w = self.make_lora_weights()
        result = apply_lora_gershgorin_decay(w, lambda_base=0.1, penalty=0.5)
        assert set(result.keys()) == set(w.keys())

    def test_apply_lora_gershgorin_decay_reduces_magnitude(self):
        w = self.make_lora_weights()
        before = sum((t ** 2).sum() for t in w.values())
        result = apply_lora_gershgorin_decay(w, lambda_base=0.1, penalty=0.5)
        after = sum((t ** 2).sum() for t in result.values())
        assert after < before

    def test_unmatched_lora_only_uniform_decay(self):
        """A weight with no matching A/B pair gets uniform decay."""
        w = {
            "base.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(16, 64),
            # No matching B
            "some_other.weight": torch.randn(10, 10),
        }
        result = apply_lora_gershgorin_decay(w, lambda_base=0.1, penalty=1.0)
        assert set(result.keys()) == set(w.keys())
        assert torch.allclose(result["some_other.weight"], w["some_other.weight"] * 0.9)
