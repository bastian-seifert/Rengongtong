"""Tests for dreaming.py — ConsolidationRoutine prompt generation."""
from __future__ import annotations

import random

from rengongtong.dreaming import (
    ConsolidationRoutine,
    FALLBACK_CONCEPTS,
    REFLECTIVE_TEMPLATES,
    SEED_TOKENS,
)


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
                    formatted = t.format(a="X", b="Y")
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
            assert "{a}" in t
            assert "{b}" in t

    def test_fallback_concepts_non_empty(self):
        assert len(FALLBACK_CONCEPTS) > 0
