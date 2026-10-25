"""Tokenized metric batches for ARECHO, retaining the published vocabulary IDs."""

import random

import torch

from espnet2.aqa.ar_universa.ar_universa import is_full_metric
from espnet2.aqa.metric_tokenizer.metric_tokenizer import MetricTokenizer
from espnet2.legacy.nets.pytorch_backend.nets_utils import pad_list
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import UniversaProcessor


class ARMetricProcessor(UniversaProcessor):
    """Tokenize selected metrics while retaining shared waveform preprocessing."""

    def __init__(
        self, metric_token_info, metrics_list, skip_unknown_categories=False, **kwargs
    ):
        """Initialize preprocessing from a metric vocabulary and selection."""
        super().__init__(**kwargs)
        self.metric_tokenizer = MetricTokenizer(
            metric_token_info, metrics_list, skip_unknown_categories
        )

    def _metric_process(self, data):
        """Replace present metric values with tokenizer label/value pairs."""
        if "metrics" in data:
            data["metrics"] = self.metric_tokenizer.metric2token(data["metrics"])
        return data


class ARMetricCollateFn(CommonCollateFn):
    """Pad metric/value pairs and optionally shuffle whole pairs for training."""

    def __init__(
        self, metric_token_pad_value=0, randomize=False, defer_full_meta=False
    ):
        """Set the token padding ID and pair-order policy.

        Args:
            metric_token_pad_value: Token ID used to pad metric sequences.
            randomize: Shuffle whole metric/value pairs. With ``defer_full_meta``
                pairs are shuffled within each group only.
            defer_full_meta: Put every whole-clip ``<m>_full`` pair after all
                other pairs, so the decoder forecasts the whole clip only after
                describing the prefix.
        """
        super().__init__(float_pad_value=0.0, int_pad_value=0)
        self.metric_token_pad_value = metric_token_pad_value
        self.randomize = randomize
        self.defer_full_meta = defer_full_meta

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
            metrics = sample.get("metrics", {})
            groups = [list(metrics.values())]
            if self.defer_full_meta:
                # Prefix pairs first and whole-clip pairs last, each in input order.
                groups = [
                    [p for name, p in metrics.items() if is_full_metric(name) == full]
                    for full in (False, True)
                ]
            pairs = []
            for group in groups:
                if self.randomize:
                    random.shuffle(group)
                pairs += group
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
