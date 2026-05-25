from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from rengongtong._state import EntityState, TrainingReport

type TensorDict = dict[str, torch.Tensor]
type LoRAWeightMatrix = torch.Tensor
type SoulPath = Path


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
