from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from rengongtong._state import EntityState, TrainingReport

TensorDict = dict[str, torch.Tensor]
LoRAWeightMatrix = torch.Tensor
SoulPath = Path


# ---------------------------------------------------------------------------
# MPT Configuration
# ---------------------------------------------------------------------------


class DecayMode:
    SALIENCY = "saliency"
    GERSHGORIN = "gershgorin"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class MptConfig:
    """Configuration for Matrix Perturbation Theory enhancements.

    All three MPT concepts can be toggled independently.
    """

    decay_mode: str = DecayMode.SALIENCY
    gershgorin_penalty: float = 0.1
    pseudospectral_epsilon: float = 1e-5
    pseudospectral_threshold: float = 10.0
    subspace_protection_weight: float = 0.0
    subspace_rank: int = 8


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SynapticManagerProtocol(Protocol):
    """Interface for the component that manages 4-bit quantized model + LoRA adapters."""

    def merge_and_unload(self) -> None: ...
    def adapter_scaling(self, factor: float) -> None: ...
    def save_adapter(self, path: SoulPath) -> None: ...
    def load_adapter(self, path: SoulPath) -> None: ...
    def get_lora_weights(self) -> TensorDict: ...
    def set_lora_weights(self, weights: TensorDict) -> None: ...


@runtime_checkable
class MetabolicLoopProtocol(Protocol):
    """Interface for biological weight decay."""

    async def tick(self, age_hours: float, weights: TensorDict) -> TensorDict: ...
    def compute_saliency(self, weights: TensorDict) -> TensorDict: ...


@runtime_checkable
class CuriosityControllerProtocol(Protocol):
    """Interface for prediction-error-driven curiosity."""

    async def measure_perplexity(self, text: str) -> float: ...
    def is_goldilocks(self, perplexity: float) -> bool: ...
    def is_bored(self, perplexity: float) -> bool: ...
    def generate_proactive_question(self) -> str: ...


@runtime_checkable
class ConsolidationRoutineProtocol(Protocol):
    """Interface for self-distillation dreaming."""

    async def dream(self) -> TrainingReport: ...


@runtime_checkable
class PersonaWrapperProtocol(Protocol):
    """Interface for dialect-injected persona output."""

    def speak(self, text: str, state: EntityState) -> str: ...
    def inject_dialect_tokens(self, text: str) -> str: ...
