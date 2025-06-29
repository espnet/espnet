# score_runner.py
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.inference.abs_metrics import AbsMetrics


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
    Validate existence and consistency of SCP files for a given test set.

    This function checks that all required SCP files exist under `decode_dir/test_name`,
    parses their contents, and verifies that all SCP files share the same set of
    `utt_id`s.

    Args:
        decode_dir (Path): Root directory containing decode results.
        test_name (str): Subdirectory name corresponding to the current test set.
        inputs (Union[List[str], Dict[str, str]]): SCP keys to read.
            - If List[str]: keys are both alias and filename.
            - If Dict[str, str]: alias → filename map.
        file_suffix (str, optional): File extension suffix. Default is ".scp".

    Raises:
        AssertionError:
            - If a required SCP file is missing
            - If any file has no entries
            - If there is a mismatch in `utt_id`s across SCP files

    Example:
        >>> validate_scp_files(
        ...     decode_dir=Path("decode"),
        ...     test_name="test-clean",
        ...     inputs={"ref": "text", "hyp": "hypothesis"},
        ... )
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
        file_path = task_dir / f"{fname}{file_suffix}"
        assert file_path.exists(), f"Missing SCP file: {file_path}"

        data = read_scp(file_path)
        assert len(data) > 0, f"No entries in {file_path}"

        key_to_data[alias] = data
        uid_sets.append(set(data.keys()))

    ref_uids = uid_sets[0]
    for i, uids in enumerate(uid_sets[1:], start=1):
        assert uids == ref_uids, f"Mismatch in UID sets across inputs, {uids, ref_uids}"


def get_class_path(obj) -> str:
    return f"{obj.__module__}.{obj.__class__.__name__}"


def load_scp_fields(
    decode_dir: Path,
    test_name: str,
    inputs: Union[List[str], Dict[str, str]],
    file_suffix: str = ".scp",
) -> Dict[str, List[str]]:
    """
    Load and align SCP files into a field-wise dictionary for evaluation.

    This function reads all required SCP files from the given test set,
    validates their consistency, and returns a dictionary with:
        - "utt_id": list of sorted utterance IDs
        - <alias>: list of aligned values for each key (e.g., "ref", "hyp")

    Args:
        decode_dir (Path): Root directory of decode results.
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
        ...     Path("decode"),
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
        - Useful as direct input to `AbsMetrics.__call__()`.
    """
    task_dir = decode_dir / test_name
    assert task_dir.exists(), f"Missing decode output: {task_dir}"

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


class ScoreRunner:
    """
    ScoreRunner evaluates inference results using user-defined metrics.

    This class is designed to work in conjunction with `InferenceRunner` and
    Kaldi-style `.scp` outputs. It loads the output files generated from decoding,
    checks their validity, and applies one or more evaluation metrics (e.g., WER, CER)
    to compute scores per test set.

    ## Usage Overview:
        1. Instantiate ScoreRunner with the evaluation config and decode directory.
        2. Metrics are automatically validated and associated with applicable test sets.
        3. Call `run()` to compute all metrics.
        4. Results are saved as a nested JSON file: `decode_dir/scores.json`.

    ## Arguments:
        config (DictConfig):
            Full configuration, including:
            - dataset.test: List of test set definitions (each with `name`)
            - metrics: List of metric configurations
                (each must have `_target_`, `inputs`, and optional `apply_to`)
        decode_dir (Path):
            Directory containing inference results organized as:
                decode_dir/
                ├── test-clean/
                │   ├── hypothesis.scp
                │   ├── text.scp
                │   └── ...
                └── test-other/
                    ├── hypothesis.scp
                    ├── text.scp
                    └── ...

    ## Metric Configuration:
        Each metric config must define:
            - _target_: Class path to a subclass of `AbsMetrics`
            - inputs: List or Dict specifying input .scp keys to use
                (e.g., ["ref", "hypothesis"])
            - apply_to (optional): List of test set names this metric applies to

        Example:
        ```yaml
        metrics:
          - _target_: my.metrics.WER
            inputs:
              - ref
              - hypothesis
            apply_to:
              - test-clean
              - test-other
        ```

    ## .scp File Requirements:
        - Must exist under `decode_dir/test_name/` with correct names.
        - All .scp files for a given metric must contain the same set of utterance IDs.
        - Text-based values are expected (e.g., for WER/CER inputs).

    ## Example:
        >>> runner = ScoreRunner(config, Path("exp/decode"))
        >>> results = runner.run()
        >>> print(results["my.metrics.WER"]["test-clean"]["wer"])  # 0.123

    ## Output:
        - The results from all metric evaluations are saved in:
              decode_dir/scores.json
        - Format:
            {
              "my.metrics.WER": {
                "test-clean": {"wer": 0.12, "sub": 5, ...},
                "test-other": {"wer": 0.18, ...}
              },
              ...
            }

    ## Notes:
        - Metric instantiation is handled via Hydra's `instantiate()`.
        - All metrics must inherit from `AbsMetrics`, implementing `__call__`.
        - Metrics are applied only to test sets listed in their `apply_to`
            (if specified).
        - This class is framework-agnostic and suitable for general-purpose
            post-evaluation.
    """

    def __init__(self, config: DictConfig, decode_dir: Path):
        self.config = config
        self.decode_dir = Path(decode_dir)
        self.test_sets = config.dataset.test
        self.metric_cfgs = config.metrics

        for test_conf in self.test_sets:
            test_name = test_conf.name
            for metric_cfg in self.metric_cfgs:
                apply_to = metric_cfg.get("apply_to", None)
                if apply_to is not None and test_name not in apply_to:
                    continue

                if not hasattr(metric_cfg, "_target_"):
                    raise KeyError(
                        f"Metric configuration must have '_target_' field to"
                        " specify the metric class."
                    )
                if not hasattr(metric_cfg, "inputs"):
                    raise ValueError(
                        f"Metric {metric_cfg._target_} must define 'inputs' field"
                        " to specify required SCP files as a list."
                    )

                inputs = OmegaConf.to_container(metric_cfg.inputs, resolve=True)
                validate_scp_files(
                    decode_dir=self.decode_dir,
                    test_name=test_name,
                    inputs=inputs,
                    file_suffix=".scp",
                )

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all scoring metrics on specified test datasets and save results to disk.

        This method evaluates the decoding results stored in `decode_dir` by applying
        user-defined metrics specified under `config.metrics`. Each metric is applied
        to the test sets listed in its `apply_to` field. The results are written to a
        JSON file and returned as a nested dictionary.

        Workflow:
            For each metric:
              - Check which test sets it applies to (via `apply_to`)
              - Load required .scp files (e.g., hypothesis.scp, ref.scp)
                from each test set
              - Pass the loaded data into the metric's `__call__` method
              - Collect and store the returned score

        Returns:
            Dict[str, Dict[str, Any]]:
                A nested dictionary of the form:
                {
                    "module.MetricClass": {
                        "test-clean": {"wer": 0.12, "insertions": 5, ...},
                        "test-other": {"wer": 0.18, "substitutions": 7, ...}
                    },
                    ...
                }

        Raises:
            KeyError: If any metric config lacks a `_target_` field.
            ValueError: If any metric config lacks an `inputs` field.
            TypeError: If instantiated metric is not a subclass of `AbsMetrics`.
            AssertionError: If required `.scp` files are missing or contain UID
                mismatches.

        Input Specification:
            Metrics must be defined in the `metrics:` section of the config
                (e.g., `evaluate.yaml`)
            Each metric must specify:
              - `_target_`: Path to the metric class (e.g., `score.wer.WER`)
              - `inputs`: Either a `List[str]` (e.g., `["ref", "hypothesis"]`)
                          or `Dict[str, str]`
                          (e.g., `{"ref": "text", "hyp": "hypothesis"}`)
              - **kwargs: Additional parameters for the metric class
              - `apply_to`: List of test set names to apply this metric to

            Example metric entry:
                - _target_: module.path.to.WER
                  inputs:
                    - ref
                    - hyp
                  apply_to:
                    - test-clean
                    - test-other

        SCP File Format:
            For each test set (e.g., `test-clean`), required .scp files must exist in:
                decode_dir/test-clean/ref.scp
                decode_dir/test-clean/hyp.scp

            Each .scp file should have format:
                <utt_id> <text or path>

        Output:
            - JSON result is saved to: `${decode_dir}/scores.json`

        Example:
            >>> score_runner = ScoreRunner(config, Path("exp/decode"))
            >>> scores = score_runner.run()
            >>> print(scores["score.wer.WER"]["test-clean"]["wer"])  # 0.12

        Related:
            - `AbsMetrics`: Each metric must inherit from this base class and implement
                `__call__`
            - `validate_scp_files`: Used to check consistency before `run()`
            - `load_scp_fields`: Loads and aligns data to pass to metrics
        """
        results = {}

        for metric_cfg in self.metric_cfgs:
            apply_to = metric_cfg.pop("apply_to", None)
            if apply_to is None:
                continue

            metric = instantiate(metric_cfg)
            if not isinstance(metric, AbsMetrics):
                raise TypeError(f"{type(metric)} is not a valid AbsMetrics instance")

            results[get_class_path(metric)] = {}
            for test_conf in self.test_sets:
                test_name = test_conf.name
                if test_name not in apply_to:
                    continue

                inputs = OmegaConf.to_container(metric_cfg.inputs, resolve=True)
                data = load_scp_fields(
                    decode_dir=self.decode_dir,
                    test_name=test_name,
                    inputs=inputs,
                    file_suffix=".scp",
                )
                metric_result = metric(data, test_name, self.decode_dir)
                results[get_class_path(metric)].update({test_name: metric_result})

        out_path = self.decode_dir / "scores.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results
