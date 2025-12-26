"""Abstract metric interfaces for ESPnet3."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List


class AbsMetrics(ABC):
    """Abstract base class for metrics used in inference evaluation.

    Subclasses must implement `__call__` and can return any JSON-serializable result.
    """

    @abstractmethod
    def __call__(
        self, data: Dict[str, List[str]], test_name: str, decode_dir: Path
    ) -> Dict[str, float]:
        """Compute metrics for a decoded test set.

        Args:
            data (Dict[str, List[str]]): Aligned fields from decode outputs.
            test_name (str): Name of the test dataset (e.g., "test-other").
            decode_dir (Path): Root path where decode results are stored.

        Returns:
            Dict[str, float]: Computed metric result(s).
        """
        raise NotImplementedError
