"""Abstract metric interfaces for ESPnet3."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List


class AbsMetric(ABC):
    """Abstract base class for metrics used in inference evaluation.

    Subclasses must implement `__call__` and can return any JSON-serializable result.
    """

    @abstractmethod
    def __call__(
        self, data: Dict[str, List[str]], test_name: str, output_dir: Path
    ) -> Dict[str, float]:
        """Compute metrics for an inference test set.

        Args:
            data (Dict[str, List[str]]): Mapping of field names to aligned lists
                of strings. Each list has the same length and index alignment.
                For example, data expects:

                .. code-block:: python

                    {
                        "utt_id": ["utt_0001", "utt_0002"],
                        "ref": ["SHE HAD YOUR DARK SUIT", "OKAY"],
                        "hyp": ["SHE HAD YOUR DARK SUIT", "OH KAY"],
                    }

                The keys are taken from inference-time SCP files and should match
                what the concrete metric class expects (e.g., ``ref``/``hyp``).
                To add extra inputs (e.g., a ``prompt`` field), define them in
                the metrics config ``inputs`` and provide a matching SCP file:

                .. code-block:: yaml

                    metrics:
                      - metric:
                          _target_: espnet3.systems.asr.metrics.wer.WER
                          clean_types:
                        inputs:
                          ref: ref
                          hyp: hyp
                          prompt: prompt

                This loads ``inference_dir/<test_name>/prompt.scp`` into
                ``data["prompt"]`` for use by the metric.
            test_name (str): Name of the test dataset (e.g., "test-other"). This
                corresponds to the test set name defined by the data organizer.
            output_dir (Path): Root path where hypothesis/reference files are stored.

        Returns:
            Dict[str, float]: Computed metric result(s).
        """
        raise NotImplementedError
