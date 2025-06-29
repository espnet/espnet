from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class AbsMetrics(ABC):
    """
    Abstract base class for metrics used in inference evaluation.

    Subclasses must implement `__call__` and can return any JSON-serializable result.
    """

    @abstractmethod
    def __call__(
        self, data: Dict[str, List[str]], test_name: str, decode_dir: Path
    ) -> Dict[str, float]:
        """
        Args:
            decode_dir (Path): Root path where decode results are stored.
            test_name (str): Name of the test dataset (e.g., 'test-other').
            inputs (List[str]): List of keys (e.g., 'text', 'hypothesis', 'rtf').

        Returns:
            Any: Computed metric result (e.g., float, dict).
        """
        raise NotImplementedError
