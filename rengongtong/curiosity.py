from __future__ import annotations

import logging
import math

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from rengongtong._state import PerplexityReport

log = logging.getLogger(__name__)


class CuriosityController:
    """Drives agentic inquiry via prediction-error / information-gain.

    Measures perplexity against a text stream.  In the Goldilocks zone
    (mid-range perplexity) the model is in an optimal learning state.
    In boredom (low perplexity) it must seek high-entropy information
    from the user.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        goldilocks_low: float = 3.0,
        goldilocks_high: float = 20.0,
        boredom_threshold: float = 3.0,
        window_tokens: int = 512,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.goldilocks_low = goldilocks_low
        self.goldilocks_high = goldilocks_high
        self.boredom_threshold = boredom_threshold
        self.window_tokens = window_tokens

        self.last_report: PerplexityReport | None = None
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

    def should_ask_question(self, report: PerplexityReport | None = None) -> bool:
        r = report or self.last_report
        if r is None:
            return False
        return r.is_bored

    def generate_curious_question(self) -> str:
        """Generate a proactive question the model wants answered."""
        self.proactive_question_count += 1
        prompt = (
            "<|system|>Du bist neugierig und willst was Neues learnen. "
            "Stell am beste eine Frage, auf die du no koane Antwort weißt.</s>\n"
            "<|user|>Was willst du wissen?</s>\n"
            "<|assistant|>"
        )
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
