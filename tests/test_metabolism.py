"""Tests for metabolism.py — MetabolicLoop decay mathematics."""
from __future__ import annotations

import math

import torch
import pytest

from rengongtong._types import DecayMode
from rengongtong.metabolism import MetabolicLoop


class TestMetabolicLoop:
    def make_weights(self) -> dict[str, torch.Tensor]:
        """Helper: creates a simple weight dict with known values."""
        return {
            "lora_A.weight": torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
            "lora_B.weight": torch.tensor([[0.5, -0.5]]),
        }

    def test_tick_returns_same_keys(self):
        loop = MetabolicLoop(base_lambda=0.01, saliency_gamma=0.9)
        w = self.make_weights()
        result = loop.tick(w, delta_hours=1.0)
        assert set(result.keys()) == set(w.keys())

    def test_tick_gershgorin_mode_returns_same_keys(self):
        loop = MetabolicLoop(base_lambda=0.01, mode=DecayMode.GERSHGORIN)
        w = self.make_weights()
        result = loop.tick(w, delta_hours=1.0)
        assert set(result.keys()) == set(w.keys())

    def test_tick_hybrid_mode_returns_same_keys(self):
        loop = MetabolicLoop(base_lambda=0.01, mode=DecayMode.HYBRID)
        w = self.make_weights()
        result = loop.tick(w, delta_hours=1.0)
        assert set(result.keys()) == set(w.keys())

    def test_gershgorin_mode_reduces_magnitude(self):
        loop = MetabolicLoop(base_lambda=0.1, mode=DecayMode.GERSHGORIN, gershgorin_penalty=0.5)
        w = self.make_weights()
        before = MetabolicLoop.l2_norm(w)
        result = loop.tick(w, delta_hours=1.0)
        after = MetabolicLoop.l2_norm(result)
        assert after < before

    def test_tick_reduces_magnitude(self):
        loop = MetabolicLoop(base_lambda=0.1, saliency_gamma=0.0)  # high decay, no saliency protection
        w = self.make_weights()
        before_norm = MetabolicLoop.l2_norm(w)
        result = loop.tick(w, delta_hours=1.0)
        after_norm = MetabolicLoop.l2_norm(result)
        assert after_norm < before_norm

    def test_tick_zero_delta_still_has_some_decay(self):
        """A tick always applies decay — delta=0 just means λ(t)=λ₀·exp(0)=λ₀."""
        loop = MetabolicLoop(base_lambda=0.1, saliency_gamma=0.9)
        w = self.make_weights()
        before = MetabolicLoop.l2_norm(w)
        result = loop.tick(w, delta_hours=0.0)
        after = MetabolicLoop.l2_norm(result)
        assert after < before  # decay still applied

    def test_compute_saliency(self):
        loop = MetabolicLoop()
        w = self.make_weights()
        sal = loop.compute_saliency(w)
        for name in w:
            assert torch.all(sal[name] >= 0.0)
            assert torch.all(sal[name] <= 1.0 + 1e-6)

    def test_high_saliency_low_decay(self):
        """High-magnitude weights should decay less than low-magnitude ones."""
        loop = MetabolicLoop(base_lambda=0.1, saliency_gamma=0.0)
        weights = {
            "high": torch.tensor([[100.0]]),
            "low": torch.tensor([[0.01]]),
        }
        result = loop.tick(weights, delta_hours=1.0)
        high_ratio = result["high"].item() / weights["high"].item()
        low_ratio = result["low"].item() / weights["low"].item()
        assert high_ratio > low_ratio, "high-saliency weight should retain more"

    def test_l2_norm(self):
        w = {"a": torch.tensor([[3.0, 4.0]])}
        norm = MetabolicLoop.l2_norm(w)
        assert math.isclose(norm, 5.0, rel_tol=1e-5)

    def test_count_parameters(self):
        w = {
            "a": torch.zeros(3, 4),
            "b": torch.zeros(2, 2),
        }
        assert MetabolicLoop.count_parameters(w) == 3 * 4 + 2 * 2  # 16

    def test_half_life_parameter_stored(self):
        loop = MetabolicLoop(half_life_hours=48.0)
        assert loop.half_life_hours == 48.0

    def test_lambda_decreases_with_time(self):
        """lambda_t should decrease as delta_hours increases."""
        loop = MetabolicLoop(base_lambda=1.0, saliency_gamma=0.5)
        l1 = 1.0 * math.exp(-0.5 * 1.0)
        l2 = 1.0 * math.exp(-0.5 * 10.0)
        assert l2 < l1

    def test_lambda_t_smaller_for_larger_delta(self):
        """The instantaneous decay rate λ(t) decreases as time passes,
        simulating memory consolidation.  The formula is λ(t) = λ₀·exp(−γ·Δt)."""
        loop = MetabolicLoop(base_lambda=0.1, saliency_gamma=0.5)
        w = self.make_weights()
        short = loop.tick(w, delta_hours=0.1)
        long = loop.tick(w, delta_hours=24.0)
        # Larger delta → smaller λ(t) → less aggressive decay
        for name in w:
            assert torch.all(long[name].abs() >= short[name].abs())
