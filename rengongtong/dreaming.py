from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone

import torch
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from rengongtong._jinja import TEMPLATES_DIR, env, render_template
from rengongtong._spectral import subspace_rotation_loss
from rengongtong._state import TrainingReport
from rengongtong._types import TensorDict

log = logging.getLogger(__name__)

NUM_REFLECTIVE_PROMPTS = 5
DREAM_LR = 3e-5
DREAM_STEPS_PER_PAIR = 2
SEED_TOKENS = ["The", "I", "Why", "How", "What", "Wos", "Des", "Mir"]

REFLECTIVE_TEMPLATES: list[str] = []
_reflective_dir = TEMPLATES_DIR / "dreaming" / "reflective"
if _reflective_dir.exists():
    for f in sorted(_reflective_dir.iterdir()):
        if f.suffix == ".j2":
            REFLECTIVE_TEMPLATES.append(f.read_text(encoding="utf-8").strip())

FALLBACK_CONCEPTS = ["Knowledge", "Learning", "Questions", "Home", "Curiosity", "Time", "Experience"]


class ConsolidationRoutine:
    """Self-distillation 'dreaming' phase.

    At 'night' (or during idle cycles), the agent:
      1. Probes its own weights for high-activation concepts.
      2. Generates reflective prompts that relate concepts.
      3. Generates answers to those prompts.
      4. Fine-tunes on the synthetic Q&A pairs to harden synaptic links.

    When *subspace_protection_weight* > 0, a Davis-Kahan sin-Θ loss term
    is added during training to prevent the principal singular vectors
    from rotating away from the base model.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        num_prompts: int = NUM_REFLECTIVE_PROMPTS,
        subspace_protection_weight: float = 0.0,
        subspace_rank: int = 8,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.num_prompts = num_prompts
        self.subspace_protection_weight = subspace_protection_weight
        self.subspace_rank = subspace_rank
        self._base_weights: TensorDict | None = None

    def _get_lora_weights(self) -> TensorDict:
        return {
            name: param.detach().cpu().clone()
            for name, param in self._model.named_parameters()
            if "lora_" in name
        }

    def capture_base_subspace(self) -> None:
        """Snapshot current LoRA weights as the reference subspace."""
        self._base_weights = self._get_lora_weights()
        log.debug("Base subspace captured (%d LoRA tensors)", len(self._base_weights))

    @torch.inference_mode()
    def extract_concepts(self, num_concepts: int = 10) -> list[str]:
        """Extract high-activation concepts by probing hidden states."""

        concepts: set[str] = set()

        for seed in SEED_TOKENS:
            input_ids = self._tokenizer(seed, return_tensors="pt").input_ids.to(
                self._model.device
            )

            outputs = self._model(
                input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits[0, -1, :]
            top_indices = logits.topk(5).indices.tolist()
            for idx in top_indices:
                token = self._tokenizer.decode([idx]).strip()
                if token and len(token) > 2:
                    concepts.add(token)

            if len(concepts) >= num_concepts:
                break

        return list(concepts)[:num_concepts]

    def generate_reflective_prompts(self, concepts: list[str]) -> list[str]:
        """Create synthetic prompts connecting concepts."""
        if len(concepts) < 2:
            concepts = FALLBACK_CONCEPTS

        prompts: list[str] = []
        for _ in range(self.num_prompts):
            a, b = random.sample(concepts, 2)
            template = random.choice(REFLECTIVE_TEMPLATES)
            prompts.append(env.from_string(template).render(a=a, b=b))

        return prompts

    def generate_answers(self, prompts: list[str]) -> list[tuple[str, str]]:
        """Generate answers for each reflective prompt via self-generation."""
        pairs: list[tuple[str, str]] = []
        for prompt in prompts:
            full_prompt = render_template("dreaming/answer_prompt.j2", prompt=prompt)
            inputs = self._tokenizer(full_prompt, return_tensors="pt").to(
                self._model.device
            )

            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            answer = self._tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()

            pairs.append((prompt, answer))
            log.debug("Dream Q: %s  A: %s...", prompt, answer[:60])

        return pairs

    def dream(self) -> TrainingReport:
        """Run one full consolidation cycle: concept extraction -> prompts -> answers -> train."""
        t0 = time.perf_counter()

        concepts = self.extract_concepts()
        log.info("Dream: extracted %d concepts", len(concepts))

        prompts = self.generate_reflective_prompts(concepts)
        pairs = self.generate_answers(prompts)

        total_loss = 0.0
        total_steps = 0

        self._model.train()
        params = [p for p in self._model.parameters() if p.requires_grad]
        optim = AdamW(params, lr=DREAM_LR)

        # Capture reference subspace if not already set
        if self.subspace_protection_weight > 0 and self._base_weights is None:
            self.capture_base_subspace()

        for question, answer in pairs:
            text = render_template(
                "dreaming/train_text.j2",
                question=question,
                answer=answer,
            ).strip()
            tokenized = self._tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(self._model.device)
            attn_mask = tokenized.attention_mask.to(self._model.device)
            labels = input_ids.clone()
            labels[labels == self._tokenizer.pad_token_id] = -100

            eos = (input_ids[0] == self._tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos) >= 3:
                labels[0, : eos[1].item() + 1] = -100

            for _ in range(DREAM_STEPS_PER_PAIR):
                optim.zero_grad()
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # Davis-Kahan subspace protection loss
                if self.subspace_protection_weight > 0 and self._base_weights is not None:
                    current = self._get_lora_weights()
                    dk_loss = subspace_rotation_loss(
                        self._base_weights,
                        current,
                        rank=self.subspace_rank,
                    )
                    loss = loss + self.subspace_protection_weight * dk_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                total_loss += loss.item()
                total_steps += 1

        self._model.eval()
        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / total_steps if total_steps else 0.0

        report = TrainingReport(
            steps=total_steps,
            loss=avg_loss,
            duration_seconds=elapsed,
        )
        log.info(
            "Dream complete — %d pairs, %d steps, loss=%.4f, duration=%.2fs",
            len(pairs), total_steps, avg_loss, elapsed,
        )
        return report
