"""Tests for _state.py — EntityState, Mood, and report models."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rengongtong._state import EntityState, Mood, PerplexityReport, SoulMetadata, TrainingReport


class TestMood:
    def test_str(self):
        assert str(Mood.GRANTIG) == "grantig"
        assert str(Mood.NEUTRAL) == "neutral"

    def test_all_moods_have_prefixes(self):
        """Every mood should have entries in persona's FRANCONIAN_PREFIXES."""
        from rengongtong.persona import FRANCONIAN_PREFIXES

        for mood in Mood:
            assert mood in FRANCONIAN_PREFIXES, f"{mood} missing from prefixes"


class TestEntityState:
    def test_default_creation(self):
        state = EntityState()
        assert state.mood == Mood.NEUTRAL
        assert state.total_feeds == 0
        assert state.total_dreams == 0
        assert state.total_decays == 0
        assert state.curiosity_level == 0.5
        assert len(state.soul_id) == 12

    def test_age_hours_increases(self):
        state = EntityState()
        age = state.age_hours
        assert age >= 0.0
        assert isinstance(age, float)

    def test_personality_traits_defaults(self):
        state = EntityState()
        assert state.personality_traits["franconian_grumpiness"] == 0.3
        assert state.personality_traits["identity_stability"] == 0.8

    def test_to_soul_dir_format(self, tmp_path: Path):
        state = EntityState()
        d = state.to_soul_dir(tmp_path)
        assert str(d).startswith(str(tmp_path))
        assert state.soul_id in str(d)

    def test_serialize_roundtrip(self):
        state = EntityState(
            mood=Mood.CURIOUS,
            total_feeds=42,
            curiosity_level=12.5,
        )
        data = json.loads(json.dumps(state.model_dump(mode="json"), default=str))
        restored = EntityState(**data)
        assert restored.mood == Mood.CURIOUS
        assert restored.total_feeds == 42

    def test_current_soul_path_set_on_init(self):
        state = EntityState()
        assert state.current_soul_path is not None
        assert state.soul_id in state.current_soul_path


class TestPerplexityReport:
    def test_defaults(self):
        r = PerplexityReport(perplexity=5.0)
        assert r.perplexity == 5.0
        assert r.is_goldilocks is False
        assert r.is_bored is False

    def test_goldilocks_true(self):
        r = PerplexityReport(perplexity=10.0, is_goldilocks=True)
        assert r.is_goldilocks is True

    def test_bored_true(self):
        r = PerplexityReport(perplexity=1.5, is_bored=True)
        assert r.is_bored is True


class TestTrainingReport:
    def test_defaults(self):
        r = TrainingReport(steps=3, duration_seconds=1.5)
        assert r.steps == 3
        assert r.loss is None
        assert r.perplexity_after is None

    def test_full(self):
        r = TrainingReport(steps=5, loss=0.25, perplexity_after=8.4, duration_seconds=2.1)
        assert r.loss == 0.25
        assert r.perplexity_after == 8.4
        assert r.duration_seconds == 2.1


class TestSoulMetadata:
    def test_age_hours(self):
        m = SoulMetadata()
        assert m.age_hours >= 0.0

    def test_defaults(self):
        m = SoulMetadata()
        assert m.feed_count == 0
        assert m.mood == Mood.NEUTRAL
