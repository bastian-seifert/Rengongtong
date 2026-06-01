from __future__ import annotations

import logging
import math

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from rengongtong._jinja import render_template
from rengongtong._spectral import pseudospectral_sensitivity
from rengongtong._state import PerplexityReport, StabilityReport

log = logging.getLogger(__name__)


class CuriosityController:
    """Drives agentic inquiry via prediction-error / information-gain.

    Measures perplexity against a text stream.  In the Goldilocks zone
    (mid-range perplexity) the model is in an optimal learning state.
    In boredom (low perplexity) it must seek high-entropy information
    from the user.

    Also measures *pseudospectral stability* — a resolvent-style norm
    that detects when the model's hidden states are near a dangerous
    region where tiny weight perturbations cause large output swings.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        goldilocks_low: float = 3.0,
        goldilocks_high: float = 20.0,
        boredom_threshold: float = 3.0,
        window_tokens: int = 512,
        pseudospectral_epsilon: float = 1e-5,
        pseudospectral_threshold: float = 10.0,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.goldilocks_low = goldilocks_low
        self.goldilocks_high = goldilocks_high
        self.boredom_threshold = boredom_threshold
        self.window_tokens = window_tokens
        self.pseudospectral_epsilon = pseudospectral_epsilon
        self.pseudospectral_threshold = pseudospectral_threshold

        self.last_report: PerplexityReport | None = None
        self.last_stability: StabilityReport | None = None
        self.proactive_question_count = 0

    @torch.inference_mode()
    def measure(self, text: str) -> PerplexityReport:
        """Compute perplexity over a single text sample."""
        self._model.eval()

        tokenized = self._tokenizer(
            text,
            truncation=True,
            max_length=self.window_tokens,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self._model.device)
        attn_mask = tokenized.attention_mask.to(self._model.device)
        labels = input_ids.clone()
        labels[labels == self._tokenizer.pad_token_id] = -100

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
        )
        loss = outputs.loss.item()
        ppl = math.exp(loss) if loss < 50 else float("inf")

        report = PerplexityReport(
            perplexity=ppl,
            is_goldilocks=self.goldilocks_low < ppl < self.goldilocks_high,
            is_bored=ppl < self.boredom_threshold,
        )
        self.last_report = report
        return report

    @torch.inference_mode()
    def measure_stability(self, input_ids: torch.Tensor) -> StabilityReport:
        """Measure pseudospectral stability via finite perturbation.

        Injects ε-scale noise into LoRA weights, measures the logit
        divergence ‖logits - logits_perturbed‖, then removes the noise.

        A large *stability_gap* means the model is near a pseudospectral
        danger zone where it is likely to hallucinate or behave erratically.
        """
        self._model.eval()
        gap = pseudospectral_sensitivity(
            forward_fn=self._model,
            input_ids=input_ids,
            epsilon=self.pseudospectral_epsilon,
        )
        report = StabilityReport(
            stability_gap=gap.item(),
            is_unstable=gap.item() > self.pseudospectral_threshold,
        )
        self.last_stability = report
        return report

    def should_ask_question(self, report: PerplexityReport | None = None) -> bool:
        r = report or self.last_report
        if r is None:
            return False
        return r.is_bored

    def generate_curious_question(self) -> str:
        """Generate a proactive question the model wants answered."""
        self.proactive_question_count += 1
        prompt = render_template("curiosity/question.j2")
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.9,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        question = self._tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        return question
