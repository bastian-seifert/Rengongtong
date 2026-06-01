from __future__ import annotations

import logging
import math

import torch

from rengongtong._spectral import apply_lora_gershgorin_decay, compute_saliency
from rengongtong._types import DecayMode, TensorDict

log = logging.getLogger(__name__)


class MetabolicLoop:
    """Biological weight decay with configurable spectral protection.

    Three decay modes are available via `DecayMode`:

    * ``SALIENCY`` — magnitude-based protection (original behaviour).
    * ``GERSHGORIN`` — penalises rows where the Gershgorin disc radius
      exceeds the diagonal, preventing eigenvalue drift.
    * ``HYBRID`` — applies both saliency protection and Gershgorin
      off-diagonal penalties.
    """

    def __init__(
        self,
        base_lambda: float = 0.01,
        half_life_hours: float = 24.0,
        saliency_gamma: float = 0.9,
        mode: str = DecayMode.SALIENCY,
        gershgorin_penalty: float = 0.1,
    ) -> None:
        self.base_lambda = base_lambda
        self.half_life_hours = half_life_hours
        self.saliency_gamma = saliency_gamma
        self.mode = mode
        self.gershgorin_penalty = gershgorin_penalty

    def tick(self, weights: TensorDict, delta_hours: float) -> TensorDict:
        """Apply temporal decay based on *delta_hours* since last tick.

        λ(t) = λ₀ · exp(−γ · Δt)  — the forgetting curve.
        """
        lambda_t = self._effective_lambda(delta_hours)
        result: TensorDict = {}

        if self.mode is DecayMode.GERSHGORIN:
            return self._gershgorin_tick(weights, lambda_t)

        if self.mode is DecayMode.HYBRID:
            return self._hybrid_tick(weights, lambda_t)

        # Default: saliency-based decay (original behaviour)
        saliency_map = self.compute_saliency(weights)
        for name, w in weights.items():
            lambda_eff = lambda_t * (1.0 - saliency_map[name] * 0.9)
            result[name] = w * (1.0 - lambda_eff)

        log.debug(
            "Saliency decay tick: Δt=%.2fh  λ(t)=%.6f",
            delta_hours, lambda_t,
        )
        return result

    def _gershgorin_tick(self, weights: TensorDict, lambda_t: float) -> TensorDict:
        result = apply_lora_gershgorin_decay(weights, lambda_t, self.gershgorin_penalty)
        log.debug(
            "Gershgorin decay tick (LoRA-aware): λ(t)=%.6f  penalty=%.2f",
            lambda_t, self.gershgorin_penalty,
        )
        return result

    def _hybrid_tick(self, weights: TensorDict, lambda_t: float) -> TensorDict:
        saliency_map = self.compute_saliency(weights)
        gershgorin_decayed = apply_lora_gershgorin_decay(weights, 0.0, self.gershgorin_penalty)
        result: TensorDict = {}

        for name, w in gershgorin_decayed.items():
            sal = saliency_map.get(name, 0.0)
            lambda_eff = lambda_t * (1.0 - sal * 0.9)
            decay_factor = 1.0 - lambda_eff
            result[name] = w * decay_factor

        log.debug(
            "Hybrid decay tick (LoRA-aware): λ(t)=%.6f  penalty=%.2f",
            lambda_t, self.gershgorin_penalty,
        )
        return result

    def _effective_lambda(self, delta_hours: float) -> float:
        return self.base_lambda * math.exp(-self.saliency_gamma * delta_hours)

    def compute_saliency(self, weights: TensorDict) -> TensorDict:
        """Saliency = magnitude relative to layer max.

        Delegates to `_spectral.compute_saliency`.
        """
        return compute_saliency(weights)

    @staticmethod
    def l2_norm(weights: TensorDict) -> float:
        total = sum((w ** 2).sum().item() for w in weights.values())
        return math.sqrt(total)

    @staticmethod
    def count_parameters(weights: TensorDict) -> int:
        return sum(w.numel() for w in weights.values())
