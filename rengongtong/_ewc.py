from __future__ import annotations

import logging
from copy import deepcopy

import torch
from torch import Tensor

from rengongtong._types import TensorDict

log = logging.getLogger(__name__)


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for preventing catastrophic forgetting.

    Stores a diagonal Fisher Information estimate and reference parameters
    for the trainable (LoRA) weights.  The EWC penalty

        L_EWC = λ/2 · Σ_i F_i · (θ_i − θ*_i)²

    discourages the model from moving away from *reference* parameters θ*
    in directions that were important for previous tasks (high F_i).

    Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting
    in neural networks", PNAS 2017.
    """

    def __init__(self, weight: float = 0.1) -> None:
        self.weight = weight
        self._fisher: TensorDict = {}
        self._reference: TensorDict = {}

    @property
    def has_fisher(self) -> bool:
        return len(self._fisher) > 0

    @torch.enable_grad()
    def compute_fisher(
        self,
        model: torch.nn.Module,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        n_samples: int = 5,
    ) -> None:
        """Estimate Fisher Information diagonal from Monte Carlo samples.

        Computes F_i = E[(∂ log p(y|x,θ) / ∂θ_i)²] by averaging squared
        gradients over *n_samples* forward-backward passes on the same batch.
        """
        model.train()

        ref = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if "lora_" in name
        }
        self._reference = ref

        fisher: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if "lora_" in name:
                fisher[name] = torch.zeros_like(param, device="cpu")

        for _ in range(n_samples):
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            for name, param in model.named_parameters():
                if "lora_" in name and param.grad is not None:
                    fisher[name] += param.grad.detach().cpu().pow(2)

        for name in fisher:
            fisher[name] /= n_samples

        self._fisher = fisher
        model.zero_grad()
        log.info(
            "EWC Fisher computed from %d samples — %d parameter groups",
            n_samples, len(self._fisher),
        )

    def penalty(self, model: torch.nn.Module) -> Tensor:
        """Compute the EWC regularisation term.

        L_EWC = λ/2 · Σ_i F_i · (θ_i − θ*_i)²
        """
        if not self._fisher or not self._reference:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        total: Tensor | None = None
        device = next(model.parameters()).device

        for name, param in model.named_parameters():
            if "lora_" in name and name in self._fisher and name in self._reference:
                diff = param - self._reference[name].to(param.device)
                reg = (self._fisher[name].to(param.device) * diff.pow(2)).sum()
                if total is None:
                    total = reg
                else:
                    total = total + reg

        if total is None:
            return torch.tensor(0.0, device=device)

        return 0.5 * self.weight * total

    def state_dict(self) -> dict:
        return {
            "weight": self.weight,
            "fisher": {k: v.clone() for k, v in self._fisher.items()},
            "reference": {k: v.clone() for k, v in self._reference.items()},
        }

    def load_state_dict(self, data: dict) -> None:
        self.weight = data["weight"]
        self._fisher = {k: v.clone() for k, v in data["fisher"].items()}
        self._reference = {k: v.clone() for k, v in data["reference"].items()}
