"""Shared audio loading and mergeable corpus means for TSE metrics."""

import numpy as np

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract.kinds.audio import Audio


def read_audio(value):
    """Return finite float32 samples in soundfile layout, and their own rate."""
    audio = Audio.coerce(value, None)
    array = audio.array.T if audio.array.ndim == 2 else audio.array
    if array.shape[0] == 0:
        raise ValueError("Cannot score empty audio")
    return array, audio.rate


class AudioMetric(BaseMetric):
    """Aggregate scalar scores without retaining an entire evaluation corpus."""

    def _init_scores(self):
        super().__init__()
        self.add_state("sample_count", 0, "sum")
        for field in self.outputs:
            self.add_state(f"sum_{field.name}", 0.0, "sum")
            self.add_state(f"count_{field.name}", 0, "sum")

    def _add_scores(self, scores, *, skip_nan=False):
        self.sample_count += 1
        for name, value in scores.items():
            value = float(value)
            if skip_nan and np.isnan(value):
                continue
            setattr(self, f"sum_{name}", getattr(self, f"sum_{name}") + value)
            setattr(self, f"count_{name}", getattr(self, f"count_{name}") + 1)

    def compute(self):
        """Return corpus means from accumulated state; reject an empty corpus."""
        if not self.sample_count:
            raise ValueError("No scores were computed. Please check the input data.")
        return {
            f.name: getattr(self, f"sum_{f.name}") / count if count else float("nan")
            for f in self.outputs
            for count in [getattr(self, f"count_{f.name}")]
        }
