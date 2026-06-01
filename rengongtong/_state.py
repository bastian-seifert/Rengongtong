from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class Mood(enum.Enum):
    GRANTIG = "grantig"
    SCHOLARLY = "scholarly"
    CURIOUS = "curious"
    BORED = "bored"
    HUNGRY = "hungry"
    NEUTRAL = "neutral"

    def __str__(self) -> str:
        return self.value


class PerplexityReport(BaseModel):
    perplexity: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_goldilocks: bool = False
    is_bored: bool = False


class StabilityReport(BaseModel):
    stability_gap: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_unstable: bool = False


class TrainingReport(BaseModel):
    steps: int
    loss: float | None = None
    perplexity_after: float | None = None
    stability_gap: float | None = None
    duration_seconds: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SoulMetadata(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feed_count: int = 0
    dream_count: int = 0
    decay_count: int = 0
    total_tokens_seen: int = 0
    mood: Mood = Mood.NEUTRAL
    lore_summary: str = ""

    @property
    def age_hours(self) -> float:
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() / 3600


class EntityState(BaseModel):
    soul_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_feed: datetime | None = None
    last_dream: datetime | None = None
    last_decay: datetime | None = None
    total_feeds: int = 0
    total_dreams: int = 0
    total_decays: int = 0
    total_tokens_seen: int = 0
    mood: Mood = Mood.NEUTRAL
    high_attention_mode: bool = False
    curiosity_level: float = 0.5
    stability_gap: float = 0.0
    personality_traits: dict[str, float] = Field(default_factory=lambda: {
        "franconian_grumpiness": 0.3,
        "scholarly_humility": 0.5,
        "curiosity_drive": 0.5,
        "identity_stability": 0.8,
        "forgetting_rate": 0.1,
    })
    current_soul_path: str | None = None

    @property
    def age_hours(self) -> float:
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() / 3600

    def to_soul_dir(self, base: Path = Path("souls")) -> Path:
        return base / f"{self.soul_id}--{self.created_at.strftime('%Y%m%dT%H%M%S')}"

    def model_post_init(self, __context: object) -> None:
        if self.current_soul_path is None:
            self.current_soul_path = str(self.to_soul_dir())
