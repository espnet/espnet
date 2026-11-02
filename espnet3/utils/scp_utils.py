"""SCP file helpers for ESPnet3 hypothesis/reference files."""

import os
from pathlib import Path
from typing import Dict, List, Union


def check_utt_id(value) -> str:
    """Return an utterance id as the string an SCP line and a file name take.

    An SCP line is ``<id> <value>``, so the id must not be empty or hold
    whitespace, or the line reads back as a different id and value. The
    same id also names the artifact files of that utterance, so it must not
    be a path: ``../x`` or ``a/b`` would write outside the directory, and
    ``.`` or ``..`` on top of it.

    Args:
        value: The id as the dataset gave it; a string, or anything whose
            ``str()`` is the id.

    Returns:
        The id as a string.

    Raises:
        ValueError: If the id is empty, holds whitespace or a path separator,
            or is ``.`` or ``..``.
    """
    name = str(value)
    if (
        not name
        or name in (".", "..")
        or any(c.isspace() for c in name)
        or "/" in name
        or "\\" in name
        or os.sep in name
    ):
        raise ValueError(
            f"utterance id {name!r} cannot head an SCP line or name a file; it "
            "must be a plain token with no whitespace or path separators"
        )
    return name


def get_class_path(obj) -> str:
    """Return the fully qualified class path for an object.

    Args:
        obj: Instantiated Python object.

    Returns:
        str: Fully qualified ``"<module>.<ClassName>"`` path.

    Example:
        >>> class Dummy:
        ...     pass
        >>> get_class_path(Dummy())
        '__main__.Dummy'
    """
    return f"{obj.__module__}.{obj.__class__.__name__}"


def load_scp_paths(
    inference_dir: Path,
    test_name: str,
    inputs: Union[List[str], Dict[str, str]],
    file_suffix: str = ".scp",
) -> Dict[str, Path]:
    """Resolve metric input aliases to SCP file paths.

    This function does not read SCP contents. It only validates that the
    expected files exist under ``inference_dir / test_name`` and returns a
    mapping from metric input aliases to their corresponding paths.

    Args:
        inference_dir (Path): Root directory of hypothesis/reference files.
        test_name (str): Subdirectory name of the test set (e.g., ``"test-clean"``).
        inputs (Union[List[str], Dict[str, str]]): Metric input declarations.
            - ``List[str]``: alias and filename are the same.
            - ``Dict[str, str]``: alias -> filename.
        file_suffix (str, optional): File extension appended to each filename.
            Default: ``".scp"``.

    Returns:
        Dict[str, Path]:
            Mapping from metric aliases to resolved file paths, e.g.

            .. code-block:: python

                {
                    "ref": Path("infer/test-clean/ref.scp"),
                    "hyp": Path("infer/test-clean/hyp.scp"),
                }

    Raises:
        AssertionError: If the test directory or any required input file is missing.

    Example:
        >>> load_scp_paths(
        ...     Path("infer"),
        ...     "test-other",
        ...     inputs={"ref": "text", "hyp": "hypothesis"},
        ... )
        {
            "ref": Path("infer/test-other/text.scp"),
            "hyp": Path("infer/test-other/hypothesis.scp"),
        }

    Notes:
        - File contents are intentionally not loaded here.
        - Consumers such as ``BaseMetric`` may stream the files or pass the
          returned paths directly to external tools.
    """
    task_dir = inference_dir / test_name
    assert (
        task_dir.exists()
    ), f"Missing hypothesis/reference files in inference_dir: {task_dir}"

    input_map = {k: k for k in inputs} if isinstance(inputs, list) else dict(inputs)
    paths = {}
    for alias, fname in input_map.items():
        path = task_dir / f"{fname}{file_suffix}"
        assert path.exists(), f"Missing SCP file: {path}"
        paths[alias] = path
    return paths
