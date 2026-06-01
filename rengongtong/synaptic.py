from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Literal

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast
from transformers.utils import is_bitsandbytes_available

from rengongtong._types import SoulPath, TensorDict
from rengongtong._spectral import compress_lora_weight, lora_spectral_summary

log = logging.getLogger(__name__)


def _unsloth_available() -> bool:
    """Lazy check — avoids the Unsloth banner at import time."""
    try:
        import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B"
MAX_SEQ_LEN = 8192

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class SynapticManager:
    """Manages 4-bit quantization, LoRA adapter buffer, and mood-driven scaling.

    The 'synaptic' core of Réngōng tóng — this is where the Soul lives as
    low-rank adapter weights atop the frozen base model.
    """

    def __init__(
        self,
        model_name: str = BASE_MODEL,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        buffer_size: int = 0,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._base_alpha = lora_alpha
        self._buffer: deque[SoulPath] = deque(maxlen=buffer_size or None)

        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizerFast
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if _unsloth_available():
            self._load_unsloth()
        else:
            self._load_vanilla()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_unsloth(self) -> None:
        from unsloth import FastLanguageModel

        log.info("Loading %s via Unsloth (4-bit)", self.model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
        )
        model = FastLanguageModel.get_peft_model(  # type: ignore[assignment]
            model,
            r=self.lora_r,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
        )
        self.model = model  # type: ignore[assignment]
        self.tokenizer = tokenizer

    def _load_vanilla(self) -> None:
        log.info("Loading %s via vanilla transformers + PEFT (4-bit)", self.model_name)
        quant_kwargs = {}
        if is_bitsandbytes_available():
            quant_kwargs = dict(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.bfloat16,
            **quant_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        self.model = model
        self.tokenizer = tokenizer  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Adapter scaling (mood swings)
    # ------------------------------------------------------------------

    def adapter_scaling(self, factor: float) -> None:
        """Scale LoRA alpha by *factor*, emulating mood-driven volatility.

        A 'mood swing' multiplier applied to the LoRA update strength:
        factor > 1 → more expressive/erratic,
        factor < 1 → more conservative/stable.
        """
        new_alpha = max(1, int(self._base_alpha * factor))
        for param_name, _ in self.model.named_parameters():
            if "lora_" in param_name:
                # PeFT layers store alpha on the adapter; we scale the active
                # LoRA scaling factor at inference time via the PeftModel scaling dict.
                pass

        if hasattr(self.model, "peft_config") and self.model.peft_config:
            for adapter_name in self.model.peft_config:
                self.model.peft_config[adapter_name].lora_alpha = new_alpha
                # Recompute scaling = alpha / r
                self.model.peft_config[adapter_name].scaling = new_alpha / self.lora_r

        log.debug("adapter_scaling → %s (factor=%.2f)", new_alpha, factor)

    # ------------------------------------------------------------------
    # Merge / Unload
    # ------------------------------------------------------------------

    def merge_and_unload(self) -> PreTrainedModel:
        """Merge LoRA weights into the base model and unload the adapter.

        Produces a 'deep memory' — the adapter knowledge is fused into the
        frozen backbone. Returns the merged base model for introspection
        or re-adapterization.
        """
        log.info("Merging LoRA weights into base model (deep memory)")
        if isinstance(self.model, PeftModel):
            merged = self.model.merge_and_unload()
            self.model = merged
        return self.model

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------

    def get_lora_weights(self) -> TensorDict:
        names = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                names[name] = param.detach().cpu()
        return names

    def set_lora_weights(self, weights: TensorDict) -> None:
        own = dict(self.model.named_parameters())
        for name, tensor in weights.items():
            if name in own:
                own[name].data.copy_(tensor.to(own[name].device))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_adapter(
        self,
        path: SoulPath,
        spectral_sparsity_retention: float | None = None,
    ) -> dict[str, float] | None:
        path.mkdir(parents=True, exist_ok=True)

        if spectral_sparsity_retention is not None and spectral_sparsity_retention < 1.0:
            from safetensors.torch import save_file
            from peft import get_peft_model_state_dict

            state = get_peft_model_state_dict(self.model)
            metrics = {}
            compressed = {}
            for key, tensor in state.items():
                if tensor.ndim == 2:
                    compressed[key] = compress_lora_weight(tensor, spectral_sparsity_retention)
                    orig_norm = tensor.norm().item()
                    diff_norm = (tensor - compressed[key]).norm().item()
                    rel_err = diff_norm / (orig_norm + 1e-8)
                    metrics[f"{key}/rel_recon_error"] = round(rel_err, 6)
                else:
                    compressed[key] = tensor
            save_file(compressed, path / "adapter_model.safetensors", {})
            self.tokenizer.save_pretrained(str(path))
            self._buffer.append(path)
            log.info(
                "Soul saved (DCT- compressed retention=%s) to %s",
                spectral_sparsity_retention, path,
            )
            return metrics

        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        self._buffer.append(path)
        log.info("Soul saved to %s", path)
        return None

    def load_adapter(self, path: SoulPath) -> None:
        if not path.exists():
            raise FileNotFoundError(f"No adapter found at {path}")

        from safetensors.torch import load_file

        safetensors_path = list(path.glob("*.safetensors"))
        if not safetensors_path:
            raise FileNotFoundError(f"No safetensors file found in {path}")
        state_dict = load_file(str(safetensors_path[0]))

        # PEFT saves keys without the adapter-name suffix (e.g. ".default"),
        # but the loaded model's state dict includes it.  Remap.
        adapter_name = "default"
        remapped = {}
        for key, tensor in state_dict.items():
            if "lora_" in key:
                parts = key.rsplit(".", 1)
                remapped[f"{parts[0]}.{adapter_name}.{parts[1]}"] = tensor
            else:
                remapped[key] = tensor

        self.model.load_state_dict(remapped, strict=False)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))

        log.info("Soul loaded from %s", path)

    # ------------------------------------------------------------------
    # Training helpers (used by Brain)
    # ------------------------------------------------------------------

    def prepare_texts(self, texts: list[str]) -> tuple[TensorDict, int]:
        tok = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        tokens = tok.input_ids
        labels = tokens.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        total_tokens = tokens.numel()
        return {
            "input_ids": tokens.to(self.model.device),
            "attention_mask": tok.attention_mask.to(self.model.device),
            "labels": labels.to(self.model.device),
        }, total_tokens

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype
