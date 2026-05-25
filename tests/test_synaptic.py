"""Tests for synaptic.py — SynapticManager adapter scaling and helpers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from rengongtong.synaptic import LORA_TARGET_MODULES, SynapticManager


class TestSynapticManagerAdapterScaling:
    """adapter_scaling is semi-pure (depends on model having peft_config)."""

    def test_scaling_with_mocked_peft_config(self):
        mgr = SynapticManager.__new__(SynapticManager)
        mgr._base_alpha = 32
        mgr.lora_r = 16
        mgr.model = MagicMock()
        mgr.model.peft_config = {
            "default": MagicMock(lora_alpha=32, scaling=32 / 16),
        }

        mgr.adapter_scaling(factor=2.0)
        assert mgr.model.peft_config["default"].lora_alpha == 64
        assert mgr.model.peft_config["default"].scaling == 64 / 16

    def test_scaling_clamp_min(self):
        """Factor should never produce alpha < 1."""
        mgr = SynapticManager.__new__(SynapticManager)
        mgr._base_alpha = 1
        mgr.lora_r = 16
        mgr.model = MagicMock()
        mgr.model.peft_config = {
            "default": MagicMock(lora_alpha=1, scaling=1 / 16),
        }

        mgr.adapter_scaling(factor=0.01)
        assert mgr.model.peft_config["default"].lora_alpha >= 1

    def test_scaling_no_peft_config_does_nothing(self):
        mgr = SynapticManager.__new__(SynapticManager)
        mgr._base_alpha = 32
        mgr.lora_r = 16
        mgr.model = MagicMock()
        mgr.model.peft_config = {}

        # Should not raise
        mgr.adapter_scaling(factor=0.5)

    def test_scaling_preserves_r(self):
        """Scaling should recalculate as alpha/r, not change r."""
        mgr = SynapticManager.__new__(SynapticManager)
        mgr._base_alpha = 32
        mgr.lora_r = 16
        mgr.model = MagicMock()
        mgr.model.peft_config = {
            "default": MagicMock(lora_alpha=32, scaling=32 / 16, r=16),
        }

        mgr.adapter_scaling(factor=2.0)
        # r should not change
        assert mgr.model.peft_config["default"].lora_alpha == 64
        assert mgr.model.peft_config["default"].scaling == 64 / 16


class TestSynapticManagerWeights:
    def test_get_lora_weights_empty_model(self):
        mgr = SynapticManager.__new__(SynapticManager)
        mgr.model = MagicMock()
        mgr.model.named_parameters = MagicMock(return_value=[])

        weights = mgr.get_lora_weights()
        assert weights == {}

    def test_set_lora_weights_empty(self):
        mgr = SynapticManager.__new__(SynapticManager)
        mgr.model = MagicMock()
        mgr.model.named_parameters = MagicMock(return_value=[])
        mgr.set_lora_weights({})  # Should not raise


class TestSynapticManagerPrepareTexts:
    def test_prepare_texts_returns_dict_and_count(self):
        mgr = SynapticManager.__new__(SynapticManager)
        mgr.model = MagicMock()
        mgr.model.device = torch.device("cpu")
        mgr.tokenizer = MagicMock()
        mgr.tokenizer.pad_token_id = 0
        mgr.tokenizer.eos_token_id = 1

        # Mock tokenizer output
        class FakeBatch:
            input_ids = torch.randint(1, 99, (2, 8))
            attention_mask = torch.ones(2, 8, dtype=torch.long)

            def to(self, device):
                return self

        mgr.tokenizer.side_effect = lambda *a, **kw: FakeBatch()

        batch, num_tokens = mgr.prepare_texts(["hello", "world"])
        assert isinstance(batch, dict)
        assert "input_ids" in batch
        assert "labels" in batch
        assert num_tokens > 0


class TestSynapticManagerIORaises:
    def test_load_nonexistent_raises(self):
        mgr = SynapticManager.__new__(SynapticManager)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                mgr.load_adapter(Path(tmp) / "nonexistent")


class TestSynapticManagerConstants:
    def test_lora_target_modules(self):
        assert "q_proj" in LORA_TARGET_MODULES
        assert "v_proj" in LORA_TARGET_MODULES
        assert len(LORA_TARGET_MODULES) == 7  # all linear layers
