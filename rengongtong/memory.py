from __future__ import annotations

import enum
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast

log = logging.getLogger(__name__)


class ReplayMode(enum.Enum):
    NLL = "nll"
    DISTILL = "distill"

    def __str__(self) -> str:
        return self.value


class ReplayStrategy(enum.Enum):
    FIFO = "fifo"
    IMPORTANCE = "importance"

    def __str__(self) -> str:
        return self.value


@dataclass
class Experience:
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    loss: float | None = None
    logits: Tensor | None = None


class ExperienceReplayBuffer:
    def __init__(
        self,
        capacity: int = 200,
        strategy: str = "fifo",
        mode: str = "nll",
        replay_ratio: float = 0.3,
    ) -> None:
        self.capacity = capacity
        self.strategy = ReplayStrategy(strategy)
        self.mode = ReplayMode(mode)
        self.replay_ratio = replay_ratio
        self._buffer: deque[Experience] = deque(maxlen=capacity)

    def push(self, text: str, loss: float | None = None) -> None:
        self._buffer.append(Experience(text=text, loss=loss))

    def __len__(self) -> int:
        return len(self._buffer)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "strategy": self.strategy.value,
            "mode": self.mode.value,
            "replay_ratio": self.replay_ratio,
            "experiences": [
                {"text": e.text, "timestamp": e.timestamp.isoformat(), "loss": e.loss}
                for e in self._buffer
            ],
        }

    def load_state_dict(self, data: dict) -> None:
        self.capacity = data["capacity"]
        self.strategy = ReplayStrategy(data["strategy"])
        self.mode = ReplayMode(data["mode"])
        self.replay_ratio = data["replay_ratio"]
        self._buffer.clear()
        for ed in data["experiences"]:
            self._buffer.append(
                Experience(
                    text=ed["text"],
                    timestamp=datetime.fromisoformat(ed["timestamp"]),
                    loss=ed["loss"],
                )
            )

    def _sample_indices(self, batch_size: int) -> list[int]:
        n = len(self._buffer)
        if n == 0:
            return []
        k = min(batch_size, n)
        if self.strategy == ReplayStrategy.IMPORTANCE:
            losses = [e.loss or 0.0 for e in self._buffer]
            total = sum(losses) or 1.0
            weights = torch.tensor([l / total for l in losses])
            return torch.multinomial(weights, k, replacement=False).tolist()
        else:
            return torch.randint(0, n, (k,)).tolist()

    def sample(self, batch_size: int) -> list[Experience]:
        return [self._buffer[i] for i in self._sample_indices(batch_size)]

    @torch.no_grad()
    def refresh_logits(
        self, model: torch.nn.Module, tokenizer: PreTrainedTokenizerFast
    ) -> None:
        if self.mode != ReplayMode.DISTILL:
            return
        device = getattr(model, "device", torch.device("cpu"))
        for exp in self._buffer:
            if exp.logits is not None:
                continue
            inputs = tokenizer(exp.text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            exp.logits = out.logits.detach().cpu()

    def replay_loss(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerFast,
        device: torch.device,
    ) -> tuple[Tensor, int]:
        n = len(self._buffer)
        if n < 1:
            return torch.tensor(0.0, device=device), 0

        batch_size = max(1, int(n * self.replay_ratio))
        experiences = self.sample(batch_size)
        texts = [e.text for e in experiences]

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        attn_mask = tokenized.attention_mask.to(device)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)

        if self.mode == ReplayMode.DISTILL:
            cached = [e.logits.to(device) for e in experiences if e.logits is not None]
            if len(cached) == len(experiences):
                loss = F.kl_div(
                    F.log_softmax(outputs.logits, dim=-1),
                    torch.stack(cached).softmax(dim=-1),
                    reduction="batchmean",
                )
            else:
                loss = outputs.loss
        else:
            loss = outputs.loss

        return loss, len(experiences)
