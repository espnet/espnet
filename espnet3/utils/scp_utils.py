"""SCP file helpers for ESPnet3 hypothesis/reference files."""

from pathlib import Path
from typing import Dict, List, Union


def read_scp(scp_file):
    """Read an SCP file into a key/value dictionary."""
    with open(scp_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return {
        line.split(maxsplit=1)[0].strip(): line.split(maxsplit=1)[1].strip()
        for line in lines
    }


def get_cls_path(obj) -> str:
    """Return the fully qualified class path for an object."""
    return f"{obj.__module__}.{obj.__class__.__name__}"


def load_scp_fields(
    inference_dir: Path,
    test_name: str,
    inputs: Union[List[str], Dict[str, str]],
    file_suffix: str = ".scp",
) -> Dict[str, List[str]]:
    """Load and align SCP files into a field-wise dictionary for evaluation.

    This function reads all required SCP files from the given test set,
    validates their consistency, and returns a dictionary with:
        - "utt_id": list of sorted utterance IDs
        - <alias>: list of aligned values for each key (e.g., "ref", "hyp")

    Args:
        inference_dir (Path): Root directory of hypothesis/reference files.
        test_name (str): Subdirectory name of the test set (e.g., "test-clean").
        inputs (Union[List[str], Dict[str, str]]): SCP keys to read.
            - List[str]: alias and filename are the same.
            - Dict[str, str]: alias → filename.

        file_suffix (str, optional): SCP file extension. Default: ".scp"

    Returns:
        Dict[str, List[str]]:
            {
                "utt_id": [...],
                "ref": [...],
                "hyp": [...],
                ...
            }

    Raises:
        AssertionError: If SCP files are missing or utterance IDs are inconsistent.

    Example:
        >>> load_scp_fields(
        ...     Path("infer"),
        ...     "test-other",
        ...     inputs={"ref": "text", "hyp": "hypothesis"},
        ... )
        {
            "utt_id": ["utt1", "utt2"],
            "ref": ["the cat", "a dog"],
            "hyp": ["the bat", "a log"]
        }

    Notes:
        - Utterance IDs are sorted to ensure consistent alignment.
        - Useful as direct input to `AbsMetric.__call__()`.
    """
    task_dir = inference_dir / test_name
    assert (
        task_dir.exists()
    ), f"Missing hypothesis/reference files in inference_dir: {task_dir}"

    input_map = {k: k for k in inputs} if isinstance(inputs, list) else dict(inputs)

    data_dicts = {}
    uid_sets = []

    for alias, fname in input_map.items():
        path = task_dir / f"{fname}{file_suffix}"
        assert path.exists(), f"Missing SCP file: {path}"
        d = read_scp(path)
        data_dicts[alias] = d
        uid_sets.append(set(d.keys()))

    ref_uids = uid_sets[0]
    for i, uids in enumerate(uid_sets[1:], start=1):
        assert uids == ref_uids, f"UID mismatch: {uids ^ ref_uids}"

    sorted_uids = sorted(ref_uids)
    result = {"utt_id": sorted_uids}
    for alias, d in data_dicts.items():
        result[alias] = [d[uid] for uid in sorted_uids]

    return result
