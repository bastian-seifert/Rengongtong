from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerFast

log = logging.getLogger(__name__)


class BenchmarkTask(ABC):
    """Base class for a single NLU benchmark task."""

    name: str = ""

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        max_samples: int | None = None,
    ) -> dict[str, float]:
        ...


# ---------------------------------------------------------------------------
# HellaSwag — choose the correct ending for a context
# ---------------------------------------------------------------------------


class HellaSwagEval(BenchmarkTask):
    name = "hellaswag"

    def __init__(self, split: str = "validation") -> None:
        self.split = split
        self._data = None

    def load(self) -> None:
        self._data = load_dataset("Rowan/hellaswag", split=self.split)
        log.info("HellaSwag %s: %d samples", self.split, len(self._data))

    @torch.inference_mode()
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        max_samples: int | None = None,
    ) -> dict[str, float]:
        if self._data is None:
            self.load()

        data = self._data
        if max_samples is not None:
            import random
            data = data.shuffle(seed=42).select(range(min(max_samples, len(data))))

        correct = 0
        total = 0

        for example in data:
            ctx = example["ctx"]
            endings = [example["endings"][i] for i in range(4)]

            scores = []
            for ending in endings:
                text = ctx + " " + ending
                inputs = tokenizer(text, truncation=True, return_tensors="pt")
                input_ids = inputs.input_ids.to(model.device)
                attn_mask = inputs.attention_mask.to(model.device)

                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
                nll = outputs.loss.item() * input_ids.size(1)
                scores.append(nll)

            if scores:
                predicted = int(min(range(len(scores)), key=lambda i: scores[i]))
                if predicted == int(example["label"]):
                    correct += 1
                total += 1

        acc = correct / max(total, 1)
        log.info("HellaSwag: %d/%d = %.2f%%", correct, total, acc * 100)
        return {"accuracy": round(acc, 4), "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# ARC — AI2 Reasoning Challenge (Challenge set)
# ---------------------------------------------------------------------------


class ARCEval(BenchmarkTask):
    name = "arc_challenge"

    def __init__(self, split: str = "test") -> None:
        self.split = split
        self._data = None

    def load(self) -> None:
        split_map = {"test": "test", "validation": "validation"}
        self._data = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split_map.get(self.split, "test"))
        log.info("ARC-Challenge %s: %d samples", self.split, len(self._data))

    @torch.inference_mode()
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        max_samples: int | None = None,
    ) -> dict[str, float]:
        if self._data is None:
            self.load()

        data = self._data
        if max_samples is not None:
            import random
            data = data.shuffle(seed=42).select(range(min(max_samples, len(data))))

        correct = 0
        total = 0
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        for example in data:
            question = example["question"]
            choices = example["choices"]
            answer_key = example.get("answerKey", "")

            scores = []
            for choice_text in choices["text"]:
                text = f"{question} {choice_text}"
                inputs = tokenizer(text, truncation=True, return_tensors="pt")
                input_ids = inputs.input_ids.to(model.device)
                attn_mask = inputs.attention_mask.to(model.device)

                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
                nll = outputs.loss.item() * input_ids.size(1)
                scores.append(nll)

            if scores and answer_key in label_map:
                predicted = int(min(range(len(scores)), key=lambda i: scores[i]))
                if predicted == label_map[answer_key]:
                    correct += 1
                total += 1

        acc = correct / max(total, 1)
        log.info("ARC-Challenge: %d/%d = %.2f%%", correct, total, acc * 100)
        return {"accuracy": round(acc, 4), "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# MMLU-Pro — multi-choice knowledge (subset)
# ---------------------------------------------------------------------------


class MMLUProEval(BenchmarkTask):
    name = "mmlu_pro"

    def __init__(self, split: str = "test", max_categories: int | None = 5) -> None:
        self.split = split
        self.max_categories = max_categories
        self._data = None

    def load(self) -> None:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=self.split)
        log.info("MMLU-Pro %s: %d samples across %d categories",
                 self.split, len(dataset),
                 len(set(dataset["category"])) if "category" in dataset.column_names else 1)
        self._data = dataset

    @torch.inference_mode()
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        max_samples: int | None = None,
    ) -> dict[str, float]:
        if self._data is None:
            self.load()

        data = self._data
        categories = list(set(data["category"])) if "category" in data.column_names else [None]
        if self.max_categories is not None:
            categories = categories[:self.max_categories]

        filtered = [ex for ex in data if ex.get("category") in categories]
        if max_samples is not None:
            import random
            filtered = random.Random(42).sample(filtered, min(max_samples, len(filtered)))

        correct = 0
        total = 0
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}

        for example in filtered:
            question = example.get("question", "")
            options = example.get("options", [])
            answer = example.get("answer", "")

            scores = []
            for opt_text in options:
                text = f"{question} {opt_text}"
                inputs = tokenizer(text, truncation=True, return_tensors="pt")
                input_ids = inputs.input_ids.to(model.device)
                attn_mask = inputs.attention_mask.to(model.device)

                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
                nll = outputs.loss.item() * input_ids.size(1)
                scores.append(nll)

            if scores and str(answer) in label_map:
                predicted = int(min(range(len(scores)), key=lambda i: scores[i]))
                if predicted == label_map[str(answer)]:
                    correct += 1
                total += 1

        acc = correct / max(total, 1)
        log.info("MMLU-Pro: %d/%d = %.2f%%", correct, total, acc * 100)
        return {"accuracy": round(acc, 4), "correct": correct, "total": total}
