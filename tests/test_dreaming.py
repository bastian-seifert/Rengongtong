"""Tests for dreaming.py — ConsolidationRoutine prompt generation."""
from __future__ import annotations

import random
from unittest.mock import MagicMock

import pytest
import torch

from rengongtong.dreaming import (
    ConsolidationRoutine,
    FALLBACK_CONCEPTS,
    REFLECTIVE_TEMPLATES,
    SEED_TOKENS,
)
from rengongtong._jinja import env


class TestConsolidationRoutinePrompts:
    """Tests for the concept → prompt pipeline (no model needed)."""

    def setup_method(self):
        self.routine = ConsolidationRoutine.__new__(ConsolidationRoutine)
        self.routine.num_prompts = 5

    def test_generate_reflective_prompts_with_enough_concepts(self):
        concepts = ["Wissen", "Lernen", "Zeit", "Hoamat", "Neigier"]
        random.seed(42)
        prompts = self.routine.generate_reflective_prompts(concepts)
        assert len(prompts) == self.routine.num_prompts
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_generate_reflective_prompts_uses_templates(self):
        concepts = ["A", "B"]
        random.seed(0)
        prompts = self.routine.generate_reflective_prompts(concepts)
        for prompt in prompts:
            # Every template has {a} and {b} replaced
            assert "{a}" not in prompt
            assert "{b}" not in prompt
            assert "A" in prompt or "B" in prompt

    def test_fallback_concepts_when_fewer_than_two(self):
        prompts = self.routine.generate_reflective_prompts(["only_one"])
        for prompt in prompts:
            # Should contain at least one FALLBACK concept
            assert any(c in prompt for c in FALLBACK_CONCEPTS)

    def test_empty_concepts_list_uses_fallback(self):
        prompts = self.routine.generate_reflective_prompts([])
        for prompt in prompts:
            assert any(c in prompt for c in FALLBACK_CONCEPTS)

    def test_all_templates_used_over_many_calls(self):
        concepts = ["X", "Y"]
        seen_templates = set()
        for seed in range(50):
            random.seed(seed)
            prompts = self.routine.generate_reflective_prompts(concepts)
            for p in prompts:
                for t in REFLECTIVE_TEMPLATES:
                    formatted = env.from_string(t).render(a="X", b="Y")
                    if formatted == p:
                        seen_templates.add(t)
        # At least 2 different templates should have been used
        assert len(seen_templates) >= 2

    def test_prompt_structure(self):
        concepts = ["Wissen", "Lernen"]
        random.seed(7)
        prompts = self.routine.generate_reflective_prompts(concepts)
        for p in prompts:
            # All templates are questions (start with Wie, Wos, Wia, Warum, Welche)
            assert p[0].isupper()

    def test_num_prompts_respected(self):
        self.routine.num_prompts = 3
        prompts = self.routine.generate_reflective_prompts(["A", "B"])
        assert len(prompts) == 3

        self.routine.num_prompts = 0
        prompts = self.routine.generate_reflective_prompts(["A", "B"])
        assert len(prompts) == 0


class TestConsolidationRoutineConstants:
    def test_seed_tokens_non_empty(self):
        assert len(SEED_TOKENS) > 0
        assert all(isinstance(t, str) for t in SEED_TOKENS)

    def test_reflective_templates_non_empty(self):
        assert len(REFLECTIVE_TEMPLATES) > 0
        for t in REFLECTIVE_TEMPLATES:
            assert "{{ a }}" in t
            assert "{{ b }}" in t

    def test_fallback_concepts_non_empty(self):
        assert len(FALLBACK_CONCEPTS) > 0


class TestConsolidationRoutineSubspaceProtection:
    """Tests for Davis-Kahan subspace protection configuration."""

    def test_default_no_subspace_protection(self):
        routine = ConsolidationRoutine.__new__(ConsolidationRoutine)
        routine.subspace_protection_weight = 0.0
        routine.subspace_rank = 8
        routine._base_weights = None
        assert routine.subspace_protection_weight == 0.0
        assert routine.subspace_rank == 8
        assert routine._base_weights is None

    def test_subspace_protection_enabled(self):
        routine = ConsolidationRoutine.__new__(ConsolidationRoutine)
        routine.subspace_protection_weight = 0.1
        routine.subspace_rank = 4
        assert routine.subspace_protection_weight == 0.1
        assert routine.subspace_rank == 4

    def test_capture_base_subspace_no_lora_params(self):
        model = MagicMock()
        model.named_parameters = MagicMock(return_value=[])
        routine = ConsolidationRoutine.__new__(ConsolidationRoutine)
        routine._model = model
        routine._base_weights = None
        routine.capture_base_subspace()
        assert routine._base_weights == {}

    def test_capture_base_subspace_with_lora_params(self):
        param = torch.nn.Parameter(torch.randn(3, 3))
        model = MagicMock()
        model.named_parameters = MagicMock(
            return_value=[("lora_A.weight", param)]
        )
        routine = ConsolidationRoutine.__new__(ConsolidationRoutine)
        routine._model = model
        routine._base_weights = None
        routine.capture_base_subspace()
        assert routine._base_weights is not None
        assert "lora_A.weight" in routine._base_weights

    def test_get_lora_weights_empty(self):
        model = MagicMock()
        model.named_parameters = MagicMock(return_value=[])
        routine = ConsolidationRoutine.__new__(ConsolidationRoutine)
        routine._model = model
        weights = routine._get_lora_weights()
        assert weights == {}

    def test_get_lora_weights_filters_non_lora(self):
        model = MagicMock()
        model.named_parameters = MagicMock(
            return_value=[
                ("lora_A.weight", torch.nn.Parameter(torch.randn(2, 2))),
                ("model.weight", torch.nn.Parameter(torch.randn(2, 2))),
            ]
        )
        routine = ConsolidationRoutine.__new__(ConsolidationRoutine)
        routine._model = model
        weights = routine._get_lora_weights()
        assert "lora_A.weight" in weights
        assert "model.weight" not in weights
