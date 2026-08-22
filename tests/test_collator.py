"""Tests for CompletionOnlyCollator dynamic padding."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from project.preprocessing import CompletionOnlyCollator


class TestCompletionOnlyCollator:
    def test_pads_variable_length_sequences(self) -> None:
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.encode = MagicMock(return_value=[99])

        collator = CompletionOnlyCollator(tokenizer, response_template="<assistant>")

        features = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
            {"input_ids": [4, 5, 6, 7, 8], "labels": [4, 5, 6, 7, 8]},
        ]
        batch = collator(features)

        assert batch["input_ids"].shape == (2, 5)
        assert batch["attention_mask"].tolist() == [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ]
        assert batch["labels"].tolist()[0][3:] == [-100, -100]
        assert batch["labels"].tolist()[1] == [4, 5, 6, 7, 8]
