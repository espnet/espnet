from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union


class AbsMetrics(ABC):
    """
    Abstract base class for metrics used in inference evaluation.

    Subclasses must implement `__call__` and can return any JSON-serializable result.
    """

    @abstractmethod
    def __call__(self, decode_dir: Path, test_name: str, inputs: List[str]) -> Any:
        """
        Args:
            decode_dir (Path): Root path where decode results are stored.
            test_name (str): Name of the test dataset (e.g., 'test-other').
            inputs (List[str]): List of keys (e.g., 'text', 'hypothesis', 'rtf').

        Returns:
            Any: Computed metric result (e.g., float, dict).
        """
        raise NotImplementedError


def read_scp(scp_file):
    with open(scp_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return {
        line.split(maxsplit=1)[0].strip(): line.split(maxsplit=1)[1].strip()
        for line in lines
    }


def validate_scp_files(
    decode_dir: Path,
    test_name: str,
    inputs: Union[List[str], Dict[str, str]],
    file_suffix: str = ".scp",
) -> Dict[str, Dict[str, str]]:
    """
    Validate and load SCP files from decode_dir/test_name.

    Supports both:
    - List[str]: ["text", "hypothesis"] → keys are same as filenames
    - Dict[str, str]: {"ref": "text", "hyp": "hypothesis"} → keys are custom

    Returns:
        Dict[str, Dict[str, str]]: {alias: {uid: value}} dict
    """
    task_dir = decode_dir / test_name
    assert task_dir.exists(), f"Missing decode output: {task_dir}"

    if isinstance(inputs, list):
        input_map = {k: k for k in inputs}
    else:
        input_map = dict(inputs)

    key_to_data: Dict[str, Dict[str, str]] = {}
    uid_sets: List[Set[str]] = []

    for alias, fname in input_map.items():
        file_path = task_dir / f"{test_name}.{fname}{file_suffix}"
        assert file_path.exists(), f"Missing SCP file: {file_path}"

        data = read_scp(file_path)
        assert len(data) > 0, f"No entries in {file_path}"

        key_to_data[alias] = data
        uid_sets.append(set(data.keys()))

    ref_uids = uid_sets[0]
    for i, uids in enumerate(uid_sets[1:], start=1):
        assert uids == ref_uids, f"Mismatch in UID sets across inputs, {uids, ref_uids}"

    return key_to_data
