"""Tokenized metric batches for ARECHO, retaining the published vocabulary IDs."""

import random

import torch

from espnet2.legacy.nets.pytorch_backend.nets_utils import pad_list
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import UniversaProcessor
from espnet2.universa.metric_tokenizer.metric_tokenizer import MetricTokenizer


class ARMetricProcessor(UniversaProcessor):
    """Tokenize selected metrics while retaining shared waveform preprocessing."""

    def __init__(self, metric_token_info, metrics_list, **kwargs):
        """Initialize preprocessing from a metric vocabulary and selection."""
        super().__init__(**kwargs)
        self.metric_tokenizer = MetricTokenizer(metric_token_info, metrics_list)

    def _metric_process(self, data):
        """Replace present metric values with tokenizer label/value pairs."""
        if "metrics" in data:
            data["metrics"] = self.metric_tokenizer.metric2token(data["metrics"])
        return data


class ARMetricCollateFn(CommonCollateFn):
    """Pad metric/value pairs and optionally shuffle whole pairs for training."""

    def __init__(self, metric_token_pad_value=0, randomize=False):
        """Set the token padding ID and pair-order policy."""
        super().__init__(float_pad_value=0.0, int_pad_value=0)
        self.metric_token_pad_value = metric_token_pad_value
        self.randomize = randomize

    def __call__(self, data):
        """Collate features and variable-length metric targets independently."""
        data = list(data)
        keys, batch = super().__call__(
            [
                (uid, {k: v for k, v in sample.items() if k != "metrics"})
                for uid, sample in data
            ]
        )
        sequences = []
        for _, sample in data:
            pairs = list(sample.get("metrics", {}).values())
            if self.randomize:
                random.shuffle(pairs)
            sequences.append(
                torch.tensor(
                    [token for pair in pairs for token in pair], dtype=torch.long
                )
            )
        batch["metrics"] = {
            "metric_token": pad_list(sequences, self.metric_token_pad_value),
            "metric_token_lengths": torch.tensor([s.numel() for s in sequences]),
        }
        return keys, batch
