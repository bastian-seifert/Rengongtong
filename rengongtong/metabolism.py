from __future__ import annotations

import logging
import math

import torch

from rengongtong._types import LoRAWeightMatrix, TensorDict

log = logging.getLogger(__name__)


class MetabolicLoop:
    """Biological weight decay with saliency-preserving selective forgetting.

    Knowledge has a half-life.  Every tick applies a decay factor λ(t) that
    depends on the time elapsed since the last decay, and a per-weight
    saliency that preserves core identity over noise.
    """

    def __init__(
        self,
        base_lambda: float = 0.01,
        half_life_hours: float = 24.0,
        saliency_gamma: float = 0.9,
    ) -> None:
        self.base_lambda = base_lambda
        self.half_life_hours = half_life_hours
        self.saliency_gamma = saliency_gamma

    def tick(self, weights: TensorDict, delta_hours: float) -> TensorDict:
        """Apply temporal decay based on *delta_hours* since last tick.

        λ(t) = λ₀ · exp(−γ · Δt)  — the forgetting curve.
        High-saliency weights get a lower effective λ.
        """
        lambda_t = self.base_lambda * math.exp(-self.saliency_gamma * delta_hours)
        saliency_map = self.compute_saliency(weights)

        result: TensorDict = {}
        for name, w in weights.items():
            s = saliency_map[name]
            λ_eff = lambda_t * (1.0 - s * 0.9)
            decay_factor = 1.0 - λ_eff
            result[name] = w * decay_factor

        log.debug(
            "Decay tick: Δt=%.2fh  λ(t)=%.6f  (effective λ range: %.6f–%.6f)",
            delta_hours, lambda_t,
            lambda_t * 0.1, lambda_t,
        )
        return result

    def compute_saliency(self, weights: TensorDict) -> TensorDict:
        """Saliency = magnitude relative to layer max.

        Weights with high absolute magnitude are deemed 'core' and
        are protected from decay.
        """
        return {
            name: w.abs() / (w.abs().max() + 1e-8)
            for name, w in weights.items()
        }

    @staticmethod
    def l2_norm(weights: TensorDict) -> float:
        total = sum((w ** 2).sum().item() for w in weights.values())
        return math.sqrt(total)

    @staticmethod
    def count_parameters(weights: TensorDict) -> int:
        return sum(w.numel() for w in weights.values())
