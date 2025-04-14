import os
import json
import torch
import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from omegaconf import DictConfig
from typing import Callable, Dict, Any, Optional, Union
from tqdm import tqdm
from hydra.utils import instantiate
from dask.distributed import get_worker, WorkerPlugin, as_completed

from espnet3 import get_espnet_model
from espnet3.data import DataOrganizer
from espnet3.parallel import set_parallel, get_client

logging.basicConfig(level=logging.INFO)

class InferencePlugin(WorkerPlugin):
    def __init__(self, model_config: Dict[str, Any], dataset):
        self.model_config = model_config
        self.dataset = dataset

    def setup(self, worker):
        model = instantiate(self.model_config)
        worker.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        model.to(worker.device)
        worker.model = model
        worker.dataset = self.dataset


def save_output_scp_format(uid: str, output: Dict[str, Any], output_dir: Path, scp_files: Dict[str, Any]):
    for key, val in output.items():
        line_parts = [uid]
        if val["type"] == "text":
            line_parts.append(val["value"])
        elif val["type"] in ("audio", "image"):
            data_dir = output_dir / "data" / key
            data_dir.mkdir(parents=True, exist_ok=True)
            file_path = data_dir / f"{uid}_{key}.{'flac' if val['type'] == 'audio' else 'png'}"
            if val["type"] == "audio":
                sf.write(file_path, val["value"], 16000, format="FLAC")
            elif val["type"] == "image":
                import imageio.v2 as imageio
                imageio.imwrite(file_path, val["value"])
            line_parts.append(str(file_path))
        scp_files[key].write(" ".join(line_parts) + "\n")


class InferenceRunner:
    """
    InferenceRunner manages test-time inference over multiple test sets defined via DataOrganizer.
    It supports both serial and parallel execution using Dask, saves outputs in Kaldi-style SCP format,
    and optionally supports resuming from partially completed runs.

    Args:
        config (Dict[str, Any]): The full configuration dictionary (from OmegaConf) that defines datasets, etc.
        model_config (Dict[str, Any]): Configuration used to instantiate the model via Hydra.
        decode_dir (Path): Directory in which outputs are saved.
        parallel (Optional[Dict[str, Any]]): Parallel configuration for Dask execution. If None, runs serially.
        resume (bool): If True, will skip any samples whose outputs already exist in .scp files.

    Raises:
        RuntimeError: If model instantiation fails or sample processing returns unexpected results.

    Notes:
        - Each test set will have one .scp file per output key (e.g., text, audio, image).
        - Audio outputs are saved as FLAC, image outputs as PNG.
        - Output dictionary returned by model must follow the format:
            {
                "text": {"type": "text", "value": "..."},
                "wav": {"type": "audio", "value": np.ndarray},
                "weight": {"type": "image", "value": np.ndarray}
            }

    Example:
        >>> runner = InferenceRunner(
        ...     config=OmegaConf.load("config.yaml"),
        ...     model_config=config.model,
        ...     decode_dir=Path("decode"),
        ...     parallel=None,
        ...     resume=True,
        ... )
        >>> runner.run()
    """
    def __init__(
        self,
        config: Dict[str, Any],
        model_config: Dict[str, Any],
        decode_dir: Path,
        parallel: Optional[Dict[str, Any]] = None,
        resume: bool = False,
    ):
        self.config = config
        self.decode_dir = Path(decode_dir)
        self.model_config = model_config
        self.parallel_config = parallel
        self.resume = resume

        self.config.dataset.train = []
        self.config.dataset.valid = []
        self.organizer: DataOrganizer = instantiate(self.config.dataset)

    def _load_completed_uids(self, scp_path: Path) -> set:
        if not scp_path.is_file():
            return set()
        with open(scp_path, "r") as f:
            return set(line.split()[0] for line in f if line.strip())

    def _run_serial(self, model):
        self.decode_dir.mkdir(parents=True, exist_ok=True)

        for test_name, dataset in self.organizer.test.items():
            logging.info(f"Processing test set: {test_name} (serial)")
            test_dir = self.decode_dir / test_name
            test_dir.mkdir(parents=True, exist_ok=True)

            scp_files = {}
            completed_uids = set()
            for key in self._collect_output_keys(dataset):
                scp_path = test_dir / f"{test_name}.{key}.scp"
                if self.resume and not completed_uids:
                    completed_uids = self._load_completed_uids(scp_path)
                scp_files[key] = open(scp_path, "a" if self.resume else "w")

            with tqdm(range(len(dataset)), desc=f"{test_name} (serial)") as pbar:
                for i in pbar:
                    uid, sample = dataset[i]
                    if uid in completed_uids:
                        continue
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"Processing UID: {uid}")
                    output = model(sample)
                    save_output_scp_format(uid, output, test_dir, scp_files)

            for f in scp_files.values():
                f.close()

    def _run_parallel(self):
        if self.parallel_config is not None:
            set_parallel(self.parallel_config)

        self.decode_dir.mkdir(parents=True, exist_ok=True)

        for test_name, dataset in self.organizer.test.items():
            logging.info(f"Processing test set: {test_name} (parallel)")
            test_dir = self.decode_dir / test_name
            test_dir.mkdir(parents=True, exist_ok=True)

            scp_files = {}
            completed_uids = set()
            for key in self._collect_output_keys(dataset):
                scp_path = test_dir / f"{test_name}.{key}.scp"
                if self.resume and not completed_uids:
                    completed_uids = self._load_completed_uids(scp_path)
                scp_files[key] = open(scp_path, "a" if self.resume else "w")

            def process(idx):
                model = get_worker().model
                dataset = get_worker().dataset
                uid, sample = dataset[idx]
                if uid in completed_uids:
                    return None
                try:
                    output = model(sample)
                except:
                    print(idx, sample, flush=True)
                    return None
                return uid, output

            with get_client(plugin=InferencePlugin(self.model_config, dataset)) as client:
                futures = client.map(process, list(range(len(dataset))))
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"{test_name} (parallel)"):
                    result = future.result()
                    if result is not None:
                        uid, output = result
                        if logging.getLogger().isEnabledFor(logging.DEBUG):
                            logging.debug(f"Processing UID: {uid}")
                        save_output_scp_format(uid, output, test_dir, scp_files)

            for f in scp_files.values():
                f.close()

    def _collect_output_keys(self, dataset):
        _, sample = dataset[0]
        model = instantiate(self.model_config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        output = model(sample)
        return list(output.keys())

    def run(self):
        if self.parallel_config is not None:
            print("Running in parallel mode")
            self._run_parallel()
        else:
            print("Running in serial mode")
            model = instantiate(self.model_config)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            self._run_serial(model)

    def compute_metrics(self, config: DictConfig) -> None:
        """
        Compute evaluation metrics per test set using config settings.

        Expects:
        - config: List of dicts, each with 'name' and 'metrics' (Hydra-style)
        - decode_dir / <test_name> / <key>.scp must exist

        Saves:
        - decode_dir / <test_name> / metrics.json (one per test set)
        - additional files (e.g., wer, wer_alignment, score) per metric class
        """
        test_names_actual = set(self.organizer.test.keys())
        test_names_config = {t.name for t in config}
        assert test_names_actual == test_names_config, (
            f"Mismatched test sets: organizer={test_names_actual}, config={test_names_config}"
        )

        for test_cfg in config:
            test_name = test_cfg.name
            out_dir = self.decode_dir / test_name
            out_dir.mkdir(parents=True, exist_ok=True)

            metric_results = {}

            for metric_cfg in test_cfg.metrics:
                metric = instantiate(metric_cfg)
                metric_name = metric.__class__.__name__
                result = metric(
                    decode_dir=self.decode_dir,
                    test_name=test_name,
                )
                metric_results[metric_name] = result

            with open(out_dir / "metrics.json", "w") as f:
                json.dump(metric_results, f, indent=2, ensure_ascii=False)

            print(f"[✓] Metrics saved for {test_name}: {list(metric_results.keys())}")
