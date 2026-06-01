from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from rengongtong._jinja import render_template
from rengongtong._state import EntityState, Mood, PerplexityReport, TrainingReport
from rengongtong._types import MptConfig, TensorDict
from rengongtong.metabolism import MetabolicLoop
from rengongtong.synaptic import SynapticManager

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEED_STEPS = 3
FEED_LR = 5e-5
DECAY_INTERVAL_MINUTES = 30
CURIOSITY_WINDOW_TOKENS = 512
DEFAULT_RESPONSE_LEN = 256


class Brain:
    """Central nervous system of Réngōng tóng.

    Owns the model, tokenizer, LoRA soul, and entity state.  Coordinates
    feeding, chatting, decay, dreaming, and curiosity — the five pillars
    of artificial life.

    Optionally integrates Matrix Perturbation Theory constructs for
    spectral stability (see `MptConfig`).
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        soul_path: Path | str | None = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        mpt: MptConfig | None = None,
    ) -> None:
        self._synapse = SynapticManager(
            model_name=model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        self.state = EntityState()
        self.mpt = mpt or MptConfig()
        self._metabolism = MetabolicLoop(
            mode=self.mpt.decay_mode,
            gershgorin_penalty=self.mpt.gershgorin_penalty,
        )
        self._running = False
        self._tasks: list[asyncio.Task] = []

        if soul_path:
            self.load_soul(Path(soul_path))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> PreTrainedModel:
        return self._synapse.model

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self._synapse.tokenizer

    @property
    def mood(self) -> Mood:
        return self.state.mood

    # ------------------------------------------------------------------
    # Feeding — incremental learning
    # ------------------------------------------------------------------

    def feed(self, text: str | list[str], steps: int = FEED_STEPS) -> TrainingReport:
        """Incrementally fine-tune on *text*.  This is how the entity 'eats'."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        t0 = time.perf_counter()
        self.model.train()

        batch, token_count = self._synapse.prepare_texts(texts)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optim = AdamW(params, lr=FEED_LR)

        total_loss = 0.0
        for step in range(steps):
            optim.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / steps
        perplexity = math.exp(avg_loss) if avg_loss < 50 else float("inf")

        self.model.eval()
        elapsed = time.perf_counter() - t0

        self.state.last_feed = datetime.now(timezone.utc)
        self.state.total_feeds += steps
        self.state.total_tokens_seen += token_count

        self._update_mood()

        report = TrainingReport(
            steps=steps,
            loss=avg_loss,
            perplexity_after=perplexity,
            duration_seconds=elapsed,
        )
        log.info(
            "Feed complete — loss=%.4f  ppl=%.2f  tokens=%d  duration=%.2fs",
            avg_loss, perplexity, token_count, elapsed,
        )
        return report

    # ------------------------------------------------------------------
    # Chatting — text generation
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        max_new_tokens: int = DEFAULT_RESPONSE_LEN,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response.  The entity speaks through its persona."""

        prompt = self._build_prompt(message, system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        reply = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return reply.strip()

    def _build_prompt(self, message: str, system_prompt: str | None = None) -> str:
        if system_prompt is None:
            system_prompt = render_template("brain/system_prompt.j2").strip()

        return render_template(
            "brain/chat.j2",
            system_prompt=system_prompt,
            message=message,
        )

    # ------------------------------------------------------------------
    # Mood heuristics
    # ------------------------------------------------------------------

    def _update_mood(self) -> None:
        if self.state.high_attention_mode:
            self.state.mood = Mood.CURIOUS
        elif self.state.total_feeds < 5:
            self.state.mood = Mood.HUNGRY
        elif self.state.curiosity_level < 0.25:
            self.state.mood = Mood.BORED
        elif self.state.stability_gap > self.mpt.pseudospectral_threshold:
            self.state.mood = Mood.BORED
        else:
            self.state.mood = Mood.GRANTIG if self.state.personality_traits["franconian_grumpiness"] > 0.6 else Mood.SCHOLARLY

    # ------------------------------------------------------------------
    # Curiosity (synchronous helper)
    # ------------------------------------------------------------------

    def measure_perplexity(self, text: str, window: int = CURIOSITY_WINDOW_TOKENS) -> PerplexityReport:
        """Compute perplexity over a sliding window of text."""
        self.model.eval()
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=window,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.model.device)
        attn_mask = tokenized.attention_mask.to(self.model.device)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss.item()

        ppl = math.exp(loss) if loss < 50 else float("inf")

        report = PerplexityReport(
            perplexity=ppl,
            is_goldilocks=3.0 < ppl < 20.0,
            is_bored=ppl < 3.0,
        )
        self.state.curiosity_level = ppl
        self._update_mood()
        return report

    # ------------------------------------------------------------------
    # Soul persistence
    # ------------------------------------------------------------------

    def save_soul(self, path: Path | None = None) -> Path:
        """Snapshot the current LoRA weights + entity state."""
        if path is None:
            path = self.state.to_soul_dir()

        self._synapse.save_adapter(path)
        with open(path / "state.json", "w") as f:
            json.dump(self.state.model_dump(), f, indent=2, default=str)

        self.state.current_soul_path = str(path)
        log.info("Soul snapshot saved to %s", path)
        return path

    def load_soul(self, path: Path) -> None:
        """Restore from a previous snapshot."""
        self._synapse.load_adapter(path)

        state_file = path / "state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            self.state = EntityState(**data)

        self.state.current_soul_path = str(path)
        log.info("Soul restored from %s", path)

    # ------------------------------------------------------------------
    # Lifecycle — async background processes
    # ------------------------------------------------------------------

    async def start_lifecycle(self) -> None:
        """Launch background tasks: metabolism, dreaming, curiosity."""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._metabolic_loop()),
        ]
        log.info("Lifecycle started: %d background task(s)", len(self._tasks))

    async def stop_lifecycle(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        log.info("Lifecycle stopped")

    async def _metabolic_loop(self) -> None:
        """Periodically apply biological weight decay via MetabolicLoop."""
        interval = DECAY_INTERVAL_MINUTES * 60
        while self._running:
            await asyncio.sleep(interval)
            weights = self._synapse.get_lora_weights()
            decayed = self._metabolism.tick(weights, delta_hours=DECAY_INTERVAL_MINUTES / 60)
            self._synapse.set_lora_weights(decayed)
            self.state.total_decays += 1
            self.state.last_decay = datetime.now(timezone.utc)
            log.debug("Metabolic decay tick applied (mode=%s)", self.mpt.decay_mode)

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def running(self) -> AsyncIterator[Brain]:
        try:
            await self.start_lifecycle()
            yield self
        finally:
            await self.stop_lifecycle()
