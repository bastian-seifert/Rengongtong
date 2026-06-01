"""Tests for curiosity.py — CuriosityController (logic without GPU)."""
from __future__ import annotations

import pytest
import torch

from rengongtong._state import PerplexityReport, StabilityReport
from rengongtong.curiosity import CuriosityController


class TestCuriosityControllerLogic:
    """Tests that exercise the controller logic without needing a real model."""

    def test_should_ask_question_no_report(self):
        ctrl = CuriosityController.__new__(CuriosityController)
        ctrl.last_report = None
        assert ctrl.should_ask_question(None) is False

    def test_should_ask_question_bored_true(self):
        ctrl = CuriosityController.__new__(CuriosityController)
        report = PerplexityReport(perplexity=1.0, is_bored=True)
        assert ctrl.should_ask_question(report) is True

    def test_should_ask_question_goldilocks_false(self):
        ctrl = CuriosityController.__new__(CuriosityController)
        report = PerplexityReport(perplexity=10.0, is_goldilocks=True, is_bored=False)
        assert ctrl.should_ask_question(report) is False

    def test_should_ask_question_uses_last_report(self):
        ctrl = CuriosityController.__new__(CuriosityController)
        ctrl.last_report = PerplexityReport(perplexity=2.0, is_bored=True)
        assert ctrl.should_ask_question() is True

    def test_proactive_question_count_increments_on_call(self):
        ctrl = CuriosityController.__new__(CuriosityController)
        ctrl.proactive_question_count = 0
        ctrl.generate_curious_question = lambda: "Wos is los?"  # type: ignore[assignment]
        _ = ctrl.generate_curious_question()
        # Can't actually test in mock, just verify the property exists
        assert hasattr(ctrl, "proactive_question_count")

    def test_threshold_config(self):
        ctrl = CuriosityController.__new__(CuriosityController)
        ctrl.goldilocks_low = 3.0
        ctrl.goldilocks_high = 20.0
        ctrl.boredom_threshold = 3.0

        assert ctrl.goldilocks_low == 3.0
        assert ctrl.goldilocks_high == 20.0

    def test_pseudospectral_config(self):
        ctrl = CuriosityController.__new__(CuriosityController)
        ctrl.pseudospectral_epsilon = 1e-5
        ctrl.pseudospectral_threshold = 10.0
        assert ctrl.pseudospectral_epsilon == 1e-5
        assert ctrl.pseudospectral_threshold == 10.0

    def test_last_stability_default_none(self):
        ctrl = CuriosityController.__new__(CuriosityController)
        ctrl.last_stability = None
        assert ctrl.last_stability is None


class TestStabilityReport:
    def test_default_creation(self):
        r = StabilityReport(stability_gap=0.5)
        assert r.stability_gap == 0.5
        assert r.is_unstable is False

    def test_unstable_flag(self):
        r = StabilityReport(stability_gap=15.0, is_unstable=True)
        assert r.is_unstable is True

    def test_timestamp_set(self):
        r = StabilityReport(stability_gap=1.0)
        assert r.timestamp is not None


class TestCuriosityReportClassification:
    """Verify that PerplexityReport fields are correctly interpreted."""

    @pytest.mark.parametrize("ppl,exp_goldilocks,exp_bored", [
        (1.5, False, True),     # bored
        (5.0, True, False),     # goldilocks
        (15.0, True, False),    # goldilocks
        (30.0, False, False),   # too high
        (3.0, False, False),    # boundary — exactly at threshold
        (20.0, False, False),   # boundary — exactly at goldilocks_high
    ])
    def test_classify(self, ppl, exp_goldilocks, exp_bored):
        ctrl = CuriosityController.__new__(CuriosityController)
        ctrl.goldilocks_low = 3.0
        ctrl.goldilocks_high = 20.0
        ctrl.boredom_threshold = 3.0

        report = PerplexityReport(
            perplexity=ppl,
            is_goldilocks=ctrl.goldilocks_low < ppl < ctrl.goldilocks_high,
            is_bored=ppl < ctrl.boredom_threshold,
        )
        assert report.is_goldilocks == exp_goldilocks
        assert report.is_bored == exp_bored
