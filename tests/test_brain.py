"""Tests for brain.py — Brain orchestrator logic and mood heuristics."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rengongtong._state import EntityState, Mood
from rengongtong._types import MptConfig, ReplayConfig
from rengongtong.brain import Brain


def _make_brain() -> Brain:
    """Helper: create a Brain instance with a working mock synapse."""
    import torch

    from rengongtong._types import MptConfig

    class FakeSynapse:
        def __init__(self):
            self.model = MagicMock()
            self.tokenizer = MagicMock()
            self.model.device = torch.device("cpu")
            self.tokenizer.pad_token_id = 0
            # Give at least one trainable param so AdamW doesn't complain
            w = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
            self.model.parameters = MagicMock(return_value=[w])
            # Make model(...) return a real loss tensor so loss.item() works
            def fake_forward(**kw):
                out = MagicMock()
                out.loss = torch.tensor(0.5, requires_grad=True)
                out.logits = torch.randn(1, 5, 100)
                return out
            self.model.side_effect = fake_forward

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
    brain.mpt = MptConfig()
    return brain


class TestBrainMood:
    """_update_mood is pure logic — no model needed."""

    def _make(self) -> Brain:
        brain = Brain.__new__(Brain)
        brain.state = EntityState()
        brain.mpt = MptConfig()
        brain._update_mood = Brain._update_mood.__get__(brain, Brain)
        return brain

    def test_high_attention_curious(self):
        brain = self._make()
        brain.state.high_attention_mode = True
        brain.state.total_feeds = 100
        brain.state.curiosity_level = 0.5
        brain._update_mood()
        assert brain.state.mood == Mood.CURIOUS

    def test_few_feeds_hungry(self):
        brain = self._make()
        brain.state.high_attention_mode = False
        brain.state.total_feeds = 3
        brain.state.curiosity_level = 0.5
        brain._update_mood()
        assert brain.state.mood == Mood.HUNGRY

    def test_low_curiosity_bored(self):
        brain = self._make()
        brain.state.high_attention_mode = False
        brain.state.total_feeds = 10
        brain.state.curiosity_level = 0.1
        brain._update_mood()
        assert brain.state.mood == Mood.BORED

    def test_grumpy_grantig(self):
        brain = self._make()
        brain.state.high_attention_mode = False
        brain.state.total_feeds = 10
        brain.state.curiosity_level = 0.5
        brain.state.personality_traits["franconian_grumpiness"] = 0.9
        brain._update_mood()
        assert brain.state.mood == Mood.GRANTIG

    def test_not_grumpy_scholarly(self):
        brain = self._make()
        brain.state.high_attention_mode = False
        brain.state.total_feeds = 10
        brain.state.curiosity_level = 0.5
        brain.state.personality_traits["franconian_grumpiness"] = 0.3
        brain._update_mood()
        assert brain.state.mood == Mood.SCHOLARLY

    def test_exactly_5_feeds_not_hungry(self):
        brain = self._make()
        brain.state.high_attention_mode = False
        brain.state.total_feeds = 5
        brain.state.curiosity_level = 0.5
        brain.state.personality_traits["franconian_grumpiness"] = 0.3
        brain._update_mood()
        assert brain.state.mood != Mood.HUNGRY

    def test_high_stability_gap_bored(self):
        brain = self._make()
        brain.mpt = MptConfig(pseudospectral_threshold=5.0)
        brain.state.high_attention_mode = False
        brain.state.total_feeds = 10
        brain.state.curiosity_level = 0.5
        brain.state.stability_gap = 10.0
        brain.state.personality_traits["franconian_grumpiness"] = 0.3
        brain._update_mood()
        assert brain.state.mood == Mood.BORED


class TestBrainMptConfig:
    def test_default_mpt_config(self):
        brain = _make_brain()
        assert brain.mpt.decay_mode == "saliency"
        assert brain.mpt.gershgorin_penalty == 0.1
        assert brain.mpt.pseudospectral_epsilon == 1e-5

    def test_custom_mpt_config(self):
        mpt = MptConfig(
            decay_mode="gershgorin",
            gershgorin_penalty=0.5,
            subspace_protection_weight=0.1,
        )
        brain = _make_brain()
        brain.mpt = mpt
        assert brain.mpt.decay_mode == "gershgorin"
        assert brain.mpt.gershgorin_penalty == 0.5
        assert brain.mpt.subspace_protection_weight == 0.1

    def test_state_has_stability_gap(self):
        from rengongtong._state import EntityState
        brain = _make_brain()
        brain.state = EntityState()
        assert hasattr(brain.state, "stability_gap")
        assert brain.state.stability_gap == 0.0

    def test_metabolism_wired_with_mpt_mode(self):
        from rengongtong.metabolism import MetabolicLoop

        brain = _make_brain()
        brain.mpt = MptConfig(decay_mode="gershgorin")
        from rengongtong._types import DecayMode

        brain._metabolism = MetabolicLoop(mode=DecayMode.GERSHGORIN)
        assert brain._metabolism.mode == "gershgorin"


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

    def test_default_system_prompt_english(self):
        brain = Brain.__new__(Brain)
        brain._build_prompt = Brain._build_prompt.__get__(brain, Brain)

        prompt = brain._build_prompt("Test")
        assert "thirst for knowledge" in prompt


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


class TestBrainReplay:
    def test_feed_with_replay_increments_count(self):
        import torch

        brain = _make_brain()
        brain.state = EntityState()
        brain._replay_buffer = None  # fresh
        replay = ReplayConfig(capacity=10, mode="nll", strategy="fifo", replay_ratio=0.3)
        from rengongtong.memory import ExperienceReplayBuffer
        brain._replay_buffer = ExperienceReplayBuffer(
            capacity=replay.capacity,
            strategy=replay.strategy,
            mode=replay.mode,
            replay_ratio=replay.replay_ratio,
        )
        # Push some prior experiences so replay has something to sample
        for i in range(5):
            brain._replay_buffer.push(f"prior_{i}", loss=0.5)

        report = brain.feed("test text")
        assert isinstance(report.loss, float)
        assert brain.state.replay_count == 1

    def test_replay_buffer_saved_in_soul_snapshot(self, tmp_path):
        brain = _make_brain()
        replay = ReplayConfig(capacity=10)
        from rengongtong.memory import ExperienceReplayBuffer
        brain._replay_buffer = ExperienceReplayBuffer(
            capacity=replay.capacity,
            strategy=replay.strategy,
            mode=replay.mode,
            replay_ratio=replay.replay_ratio,
        )
        brain._replay_buffer.push("hello", loss=0.5)
        brain.state = EntityState()

        path = tmp_path / "replay_soul"
        brain.save_soul(path)

        state_file = path / "state.json"
        assert state_file.exists()
        import json
        data = json.loads(state_file.read_text())
        assert "replay_buffer" in data
        assert data["replay_buffer"]["experiences"][0]["text"] == "hello"
