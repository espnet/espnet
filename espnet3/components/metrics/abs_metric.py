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
        self, data: Dict[str, List[str]], test_name: str, infer_dir: Path
    ) -> Dict[str, float]:
        """Compute metrics for an inference test set.

        Args:
            data (Dict[str, List[str]]): Aligned fields from hypothesis/reference files.
            test_name (str): Name of the test dataset (e.g., "test-other").
            infer_dir (Path): Root path where hypothesis/reference files are stored.

        Returns:
            Dict[str, float]: Computed metric result(s).
        """
        raise NotImplementedError
