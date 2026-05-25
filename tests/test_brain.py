"""Tests for brain.py — Brain orchestrator logic and mood heuristics."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rengongtong._state import EntityState, Mood
from rengongtong.brain import Brain


def _make_brain() -> Brain:
    """Helper: create a Brain instance with a working mock synapse."""
    import torch

    class FakeSynapse:
        def __init__(self):
            self.model = MagicMock()
            self.tokenizer = MagicMock()
            self.model.device = torch.device("cpu")
            self.tokenizer.pad_token_id = 0

        def save_adapter(self, path):
            path.mkdir(parents=True, exist_ok=True)

        def load_adapter(self, path):
            if not path.exists():
                raise FileNotFoundError(str(path))

        def prepare_texts(self, texts, **kw):
            import torch
            return {"input_ids": torch.zeros(1, 4, dtype=torch.long), "labels": torch.zeros(1, 4, dtype=torch.long), "attention_mask": torch.ones(1, 4, dtype=torch.long)}, 4

        def get_lora_weights(self):
            return {}

        def set_lora_weights(self, _):
            pass

    brain = Brain.__new__(Brain)
    brain._synapse = FakeSynapse()  # type: ignore[assignment]
    return brain


class TestBrainMood:
    """_update_mood is pure logic — no model needed."""

    def test_high_attention_curious(self):
        brain = Brain.__new__(Brain)
        brain.state = EntityState()
        brain._update_mood = Brain._update_mood.__get__(brain, Brain)

        brain.state.high_attention_mode = True
        brain.state.total_feeds = 100
        brain.state.curiosity_level = 0.5
        brain._update_mood()
        assert brain.state.mood == Mood.CURIOUS

    def test_few_feeds_hungry(self):
        brain = Brain.__new__(Brain)
        brain.state = EntityState()
        brain._update_mood = Brain._update_mood.__get__(brain, Brain)

        brain.state.high_attention_mode = False
        brain.state.total_feeds = 3
        brain.state.curiosity_level = 0.5
        brain._update_mood()
        assert brain.state.mood == Mood.HUNGRY

    def test_low_curiosity_bored(self):
        brain = Brain.__new__(Brain)
        brain.state = EntityState()
        brain._update_mood = Brain._update_mood.__get__(brain, Brain)

        brain.state.high_attention_mode = False
        brain.state.total_feeds = 10
        brain.state.curiosity_level = 0.1
        brain._update_mood()
        assert brain.state.mood == Mood.BORED

    def test_grumpy_grantig(self):
        brain = Brain.__new__(Brain)
        brain.state = EntityState()
        brain._update_mood = Brain._update_mood.__get__(brain, Brain)

        brain.state.high_attention_mode = False
        brain.state.total_feeds = 10
        brain.state.curiosity_level = 0.5
        brain.state.personality_traits["franconian_grumpiness"] = 0.9
        brain._update_mood()
        assert brain.state.mood == Mood.GRANTIG

    def test_not_grumpy_scholarly(self):
        brain = Brain.__new__(Brain)
        brain.state = EntityState()
        brain._update_mood = Brain._update_mood.__get__(brain, Brain)

        brain.state.high_attention_mode = False
        brain.state.total_feeds = 10
        brain.state.curiosity_level = 0.5
        brain.state.personality_traits["franconian_grumpiness"] = 0.3
        brain._update_mood()
        assert brain.state.mood == Mood.SCHOLARLY

    def test_exactly_5_feeds_not_hungry(self):
        brain = Brain.__new__(Brain)
        brain.state = EntityState()
        brain._update_mood = Brain._update_mood.__get__(brain, Brain)

        brain.state.high_attention_mode = False
        brain.state.total_feeds = 5
        brain.state.curiosity_level = 0.5
        brain.state.personality_traits["franconian_grumpiness"] = 0.3
        brain._update_mood()
        assert brain.state.mood != Mood.HUNGRY


class TestBrainBuildPrompt:
    def test_build_prompt_contains_system_and_user(self):
        brain = Brain.__new__(Brain)
        brain._build_prompt = Brain._build_prompt.__get__(brain, Brain)

        prompt = brain._build_prompt("Hello")
        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
        assert "Hello" in prompt

    def test_custom_system_prompt(self):
        brain = Brain.__new__(Brain)
        brain._build_prompt = Brain._build_prompt.__get__(brain, Brain)

        prompt = brain._build_prompt("Test", system_prompt="Custom sys")
        assert "Custom sys" in prompt

    def test_default_system_prompt_franconian(self):
        brain = Brain.__new__(Brain)
        brain._build_prompt = Brain._build_prompt.__get__(brain, Brain)

        prompt = brain._build_prompt("Test")
        assert "fränkischa" in prompt
        assert "Frängisch" in prompt


class TestBrainApplyDecay:
    def test_apply_decay_reduces_weights(self):
        import torch

        brain = Brain.__new__(Brain)
        brain._apply_decay = Brain._apply_decay.__get__(brain, Brain)

        weights = {
            "lora_A": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }
        result = brain._apply_decay(weights, base_lambda=0.1)
        assert torch.all(result["lora_A"] < weights["lora_A"])

    def test_apply_decay_preserves_shape(self):
        import torch

        brain = Brain.__new__(Brain)
        brain._apply_decay = Brain._apply_decay.__get__(brain, Brain)

        weights = {
            "a": torch.randn(3, 4),
            "b": torch.randn(2, 2),
        }
        result = brain._apply_decay(weights, base_lambda=0.01)
        for name in weights:
            assert result[name].shape == weights[name].shape

    def test_apply_decay_preserves_dtype(self):
        import torch

        brain = Brain.__new__(Brain)
        brain._apply_decay = Brain._apply_decay.__get__(brain, Brain)

        weights = {
            "a": torch.randn(3, 4, dtype=torch.float64),
        }
        result = brain._apply_decay(weights)
        assert result["a"].dtype == torch.float64


class TestBrainStatePersistence:
    def test_save_soul_creates_state_json(self, tmp_path: Path):
        brain = _make_brain()
        brain.state = EntityState()

        path = tmp_path / "test_soul"
        brain.save_soul(path)

        state_file = path / "state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["soul_id"] == brain.state.soul_id

    def test_save_soul_updates_current_soul_path(self, tmp_path: Path):
        brain = _make_brain()
        brain.state = EntityState()

        path = tmp_path / "test_soul"
        brain.save_soul(path)
        assert brain.state.current_soul_path == str(path)

    def test_save_soul_calls_synaptic_save(self, tmp_path: Path):
        brain = _make_brain()
        brain.state = EntityState()
        real_save = brain._synapse.save_adapter
        brain._synapse.save_adapter = MagicMock(side_effect=lambda p: p.mkdir(parents=True, exist_ok=True))

        path = tmp_path / "test_soul"
        brain.save_soul(path)
        brain._synapse.save_adapter.assert_called_once_with(path)

    def test_mood_property(self):
        brain = Brain.__new__(Brain)
        brain.state = EntityState(mood=Mood.CURIOUS)
        assert brain.mood == Mood.CURIOUS
