from __future__ import annotations

import torch

from rengongtong.memory import ExperienceReplayBuffer, ReplayMode, ReplayStrategy


class TestReplayBufferCore:
    def test_push_and_sample_returns_correct_count(self):
        buf = ExperienceReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(f"text_{i}", loss=float(i) / 10)
        sampled = buf.sample(5)
        assert len(sampled) == 5
        texts = {e.text for e in sampled}
        assert all(t.startswith("text_") for t in texts)

    def test_fifo_eviction(self):
        buf = ExperienceReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(f"text_{i}", loss=float(i))
        assert len(buf) == 5
        texts = [e.text for e in buf._buffer]
        assert all(t in texts for t in ["text_5", "text_6", "text_7", "text_8", "text_9"])
        assert "text_0" not in texts
        assert "text_4" not in texts

    def test_empty_buffer_returns_zero_loss(self, mock_model, mock_tokenizer):
        buf = ExperienceReplayBuffer(capacity=100)
        loss, n = buf.replay_loss(mock_model, mock_tokenizer, torch.device("cpu"))
        assert n == 0
        assert loss.item() == 0.0

    def test_state_dict_roundtrip(self):
        buf = ExperienceReplayBuffer(capacity=50, strategy="fifo", mode="nll", replay_ratio=0.5)
        for i in range(5):
            buf.push(f"text_{i}", loss=float(i))
        state = buf.state_dict()
        buf2 = ExperienceReplayBuffer()
        buf2.load_state_dict(state)
        assert len(buf2) == 5
        assert buf2.capacity == 50
        assert buf2.strategy == ReplayStrategy.FIFO
        assert buf2.mode == ReplayMode.NLL
        assert buf2.replay_ratio == 0.5
        for i in range(5):
            assert buf2._buffer[i].text == f"text_{i}"

    def test_replay_ratio_respected(self):
        buf = ExperienceReplayBuffer(capacity=100, replay_ratio=0.5)
        for i in range(20):
            buf.push(f"text_{i}", loss=float(i))
        loss, n = buf.replay_loss(
            type("M", (), {"device": torch.device("cpu"), "parameters": lambda: iter([]), "eval": lambda: None, "train": lambda: None, "__call__": lambda *a, **kw: type("O", (), {"loss": torch.tensor(0.5, requires_grad=True), "logits": torch.randn(1, 5, 100)})()})(),
            type("T", (), {"pad_token_id": 0, "__call__": lambda *a, **kw: type("E", (), {"input_ids": torch.zeros(1, 4, dtype=torch.long), "attention_mask": torch.ones(1, 4, dtype=torch.long)})()})(),
            torch.device("cpu"),
        )
        assert n > 0
        assert n <= 20

    def test_importance_sampling_prefers_high_loss(self, mock_model, mock_tokenizer):
        buf = ExperienceReplayBuffer(capacity=100, strategy="importance")
        for i in range(20):
            buf.push(f"text_{i}", loss=float(i) / 20.0)
        counts: dict[str, int] = {}
        for _ in range(200):
            sampled = buf.sample(5)
            for e in sampled:
                counts[e.text] = counts.get(e.text, 0) + 1
        high_loss_texts = {f"text_{i}" for i in range(15, 20)}
        low_loss_texts = {f"text_{i}" for i in range(0, 5)}
        high_total = sum(counts.get(t, 0) for t in high_loss_texts)
        low_total = sum(counts.get(t, 0) for t in low_loss_texts)
        assert high_total > low_total, (
            f"High-loss items ({high_total}) should be sampled more than low-loss ({low_total})"
        )


class TestReplayBufferNLL:
    def test_nll_replay_loss_returns_grad_tensor(self, mock_model, mock_tokenizer):
        buf = ExperienceReplayBuffer(capacity=100)
        for i in range(5):
            buf.push(f"text_{i}", loss=0.5)
        loss, n = buf.replay_loss(mock_model, mock_tokenizer, torch.device("cpu"))
        assert n == 1  # 5 * 0.3 = 1.5 -> int -> 1 (max(1, ...))
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad is True or loss.grad_fn is not None


class TestReplayBufferDistill:
    def test_refresh_logits_caches_correctly(self, mock_model, mock_tokenizer):
        buf = ExperienceReplayBuffer(capacity=10, mode="distill")
        buf.push("hello world")
        assert buf._buffer[0].logits is None
        buf.refresh_logits(mock_model, mock_tokenizer)
        assert buf._buffer[0].logits is not None
        assert buf._buffer[0].logits.device.type == "cpu"

    def test_distill_skips_if_not_distill_mode(self, mock_model, mock_tokenizer):
        buf = ExperienceReplayBuffer(capacity=10, mode="nll")
        buf.push("hello world")
        buf.refresh_logits(mock_model, mock_tokenizer)
        assert buf._buffer[0].logits is None
