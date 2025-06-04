# score_runner.py
from pathlib import Path
import json
from typing import Dict, Any, List, Union, Set
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
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
        file_path = task_dir / f"{fname}{file_suffix}"
        assert file_path.exists(), f"Missing SCP file: {file_path}"

        data = read_scp(file_path)
        assert len(data) > 0, f"No entries in {file_path}"

        key_to_data[alias] = data
        uid_sets.append(set(data.keys()))

    ref_uids = uid_sets[0]
    for i, uids in enumerate(uid_sets[1:], start=1):
        assert uids == ref_uids, f"Mismatch in UID sets across inputs, {uids, ref_uids}"

    return key_to_data

def get_class_path(obj) -> str:
    return f"{obj.__module__}.{obj.__class__.__name__}"

def load_scp_fields(
    decode_dir: Path,
    test_name: str,
    inputs: Union[List[str], Dict[str, str]],
    file_suffix: str = ".scp",
) -> Dict[str, List[str]]:
    """
    Return a dict like {"utt_id": [...], "ref": [...], "hyp": [...]}
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
    ScoreRunner loads user-defined metrics and applies them to inference outputs
    in decode_dir, filtering by `apply_to` if specified in each metric config.

    Args:
        config (DictConfig): Configuration containing dataset.test and metrics list
        decode_dir (Path): Directory where inference outputs (.scp files) are saved
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

                inputs = OmegaConf.to_container(metric_cfg.inputs, resolve=True)
                try:
                    validate_scp_files(
                        decode_dir=self.decode_dir,
                        test_name=test_name,
                        inputs=inputs,
                        file_suffix=".scp",
                    )
                except AssertionError as e:
                    raise RuntimeError(
                        f"[{test_name}] Validation failed for metric"
                        f"{metric_cfg._target_}: {e}"
                    )

    def run(self) -> Dict[str, Dict[str, Any]]:
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
                    file_suffix=".scp"
                )
                metric_result = metric(data, test_name, self.decode_dir)
                results[get_class_path(metric)].update({
                    test_name: metric_result
                })

        out_path = self.decode_dir / "scores.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results
