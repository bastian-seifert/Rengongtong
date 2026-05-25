"""Reusable fixtures for Réngōng tóng tests."""
from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, create_autospec

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast


@pytest.fixture
def any_mood_state():
    """Minimal EntityState-like duck.  Import inside fixture to avoid
    pulling in the full package at collection time."""
    from rengongtong._state import EntityState, Mood

    return EntityState(mood=Mood.NEUTRAL)


@pytest.fixture
def scholarly_state():
    from rengongtong._state import EntityState, Mood

    return EntityState(
        mood=Mood.SCHOLARLY,
        personality_traits={
            "franconian_grumpiness": 0.2,
            "scholarly_humility": 0.9,
            "curiosity_drive": 0.5,
            "identity_stability": 0.8,
            "forgetting_rate": 0.1,
        },
    )


@pytest.fixture
def grantig_state():
    from rengongtong._state import EntityState, Mood

    return EntityState(
        mood=Mood.GRANTIG,
        personality_traits={
            "franconian_grumpiness": 0.9,
            "scholarly_humility": 0.1,
            "curiosity_drive": 0.3,
            "identity_stability": 0.8,
            "forgetting_rate": 0.1,
        },
    )


@pytest.fixture
def mock_model() -> MagicMock:
    """A lightweight model mock that returns trivial loss/generation."""

    model = create_autospec(PreTrainedModel, instance=True)
    model.device = torch.device("cpu")
    model.dtype = torch.float32

    def fake_forward(input_ids=None, attention_mask=None, labels=None, **kw):
        batch = MagicMock()
        batch.loss = torch.tensor(0.5, requires_grad=True)
        batch.logits = torch.randn(1, 5, 100)
        return batch

    model.side_effect = fake_forward
    model.return_value = None

    # Make __call__ work (PreTrainedModel uses __call__ not forward directly)
    fake_call = MagicMock(side_effect=fake_forward)
    model.__call__ = model.side_effect

    def fake_generate(**kw):
        return torch.randint(0, 100, (1, 10))

    model.generate = MagicMock(side_effect=fake_generate)

    model.named_parameters = MagicMock(return_value=[])
    model.parameters = MagicMock(return_value=[])
    model.train = MagicMock()
    model.eval = MagicMock()

    return model


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """A tokenizer mock that returns simple tensor stubs."""

    tok = create_autospec(PreTrainedTokenizerFast, instance=True)
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.vocab_size = 100

    def fake_encode(text, **kw):
        return torch.randint(1, 99, (1, 8))

    def fake_call(text, **kw):
        class FakeEncoding:
            def __init__(self):
                seq_len = 8
                self.input_ids = torch.randint(1, 99, (1, seq_len))
                self.attention_mask = torch.ones(1, seq_len, dtype=torch.long)

            def to(self, device):
                return self

        return FakeEncoding()

    tok.side_effect = fake_call
    tok.encode = MagicMock(side_effect=fake_encode)
    tok.decode = MagicMock(side_effect=lambda ids, **kw: "mock token")
    tok.__call__ = MagicMock(side_effect=fake_call)

    return tok
