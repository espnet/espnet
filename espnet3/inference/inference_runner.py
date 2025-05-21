import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import psutil
import soundfile as sf
import torch
from dask.distributed import WorkerPlugin, as_completed, get_worker
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from espnet2.train.preprocessor import AbsPreprocessor
from espnet3 import get_espnet_model
from espnet3.data import (
    DatasetWithTransform, get_wrapped_transform, do_nothing_transform
)
from espnet3.parallel import get_client, set_parallel

logging.basicConfig(level=logging.INFO)


def save_output_scp_format(
    uid: str, output: Dict[str, Any], output_dir: Path, scp_files: Dict[str, Any]
):
    for key, val in output.items():
        line_parts = [uid]
        if val["type"] == "text":
            line_parts.append(val["value"])
        elif val["type"] in ("audio", "image"):
            data_dir = output_dir / "data" / key
            data_dir.mkdir(parents=True, exist_ok=True)
            file_path = (
                data_dir / f"{uid}_{key}.{'flac' if val['type'] == 'audio' else 'png'}"
            )
            if val["type"] == "audio":
                sf.write(file_path, val["value"], 16000, format="FLAC")
            elif val["type"] == "image":
                import imageio.v2 as imageio

                imageio.imwrite(file_path, val["value"])
            line_parts.append(str(file_path))
        scp_files[key].write(" ".join(line_parts) + "\n")


def read_audio(path: str) -> np.ndarray:
    wav, _ = sf.read(path)
    return wav


def stream_audio(path: str, chunk_sec: float = 0.01) -> Iterator[np.ndarray]:
    with sf.SoundFile(path, "r") as f:
        sr = f.samplerate
        frames_per_chunk = int(sr * chunk_sec)
        while True:
            chunk = f.read(frames_per_chunk, dtype="float32")
            if len(chunk) == 0:
                break
            yield chunk


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def stream_text(path: str, chunk_chars: int = 5) -> Iterator[str]:
    text = read_text(path)
    for i in range(0, len(text), chunk_chars):
        yield text[i : i + chunk_chars]


class InferencePlugin(WorkerPlugin):
    def __init__(self, inference_runner):
        self.inference_runner = inference_runner

    def setup(self, worker):
        # Set device based on worker's resources
        worker.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize model on worker
        worker.model = self.inference_runner.initialize_model(worker.device)
        # Store inference runner for access
        worker.inference_runner = self.inference_runner


class InferenceRunner:
    """
    InferenceRunner manages test-time inference over multiple test sets defined
    via DataOrganizer. It supports both serial and parallel execution using Dask,
    saves outputs in Kaldi-style SCP format, and optionally supports resuming from
    partially completed runs.

    Args:
        config (Dict[str, Any]): The full configuration dictionary (from OmegaConf)
            that defines datasets, etc.
        model_config (Dict[str, Any]): Configuration used to instantiate the model
            via Hydra.
        decode_dir (Path): Directory in which outputs are saved.
        parallel (Optional[Dict[str, Any]]): Parallel configuration for Dask execution.
            If None, runs serially.
        resume (bool): If True, will skip any samples whose outputs already exist in
            .scp files.

    Raises:
        RuntimeError: If model instantiation fails or sample processing returns
            unexpected results.

    Notes:
        - Each test set will have one .scp file per output key
            (e.g., text, audio, image).
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
        model_config: DictConfig = None,
        dataset_config: DictConfig = None,
        stream: bool = False,
        parallel: Optional[Dict[str, Any]] = None,
    ):
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.stream = stream
        self.parallel_config = parallel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.dataset = None
        self.current_dataset_key = None

    def initialize_model(self, device=None):
        return instantiate(self.model_config)

    def _initialize_model(self, device=None):
        if self.model is None:
            self.model = self.initialize_model(device)

        return self.model

    def initialize_dataset(self, dataset_key, dataset_config=None):
        if dataset_config is None:
            dataset_config = self.dataset_config

        # Look for test dataset
        test_ds_conf = None
        for ds_conf in dataset_config.test:
            if ds_conf.name == dataset_key:
                test_ds_conf = ds_conf
                break

        if test_ds_conf is None:
            RuntimeError(f"{dataset_key} not found in inference config.")

        # Preprocessor
        if hasattr(dataset_config, "preprocessor"):
            preprocessor = instantiate(dataset_config.preprocessor)
        else:
            preprocessor = do_nothing_transform

        is_espnet_preprocessor = isinstance(preprocessor, AbsPreprocessor)

        if hasattr(test_ds_conf, "transform"):
            transform = instantiate(test_ds_conf.transform)
        else:
            transform = do_nothing_transform

        wrapped_transform = get_wrapped_transform(
            is_espnet_preprocessor,
            transform,
            preprocessor
        )
        return DatasetWithTransform(instantiate(ds_conf.dataset), wrapped_transform)

    def _initialize_dataset(self, dataset_key, dataset_config=None):
        if self.current_dataset_key != dataset_key:
            self.dataset = self.initialize_dataset(dataset_key, dataset_config)

        return self.dataset

    def read(
        self,
        input_type: str,
        path: str,
        stream: bool = False,
        chunk_sec: float = 0.01,
        chunk_chars: int = 5,
    ) -> Union[np.ndarray, str, Iterator[np.ndarray], Iterator[str]]:
        if input_type == "audio":
            if stream:
                return stream_audio(path, chunk_sec)
            else:
                return read_audio(path)
        elif input_type == "text":
            if stream:
                return stream_text(path, chunk_chars)
            else:
                return read_text(path)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

    def write(self, uid: str, output: Dict[str, Any], output_dir: Union[str, Path]):
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        for key, val in output.items():
            scp_path = output_dir / f"{key}.scp"
            line_parts = [uid]
            if not isinstance(val, str) and not isinstance(val, dict):
                raise ValueError(
                    f"output should be a string or dict. Got {type(output)}"
                )

            if isinstance(val, str):
                line_parts.append(val)
            elif "type" not in val:
                raise ValueError(
                    f"Please specify type of value. 'text' or 'audio' is supported."
                )
            elif val["type"] == "text":
                line_parts.append(val["value"])
            elif val["type"] == "audio":
                data_dir = output_dir / "data" / key
                data_dir.mkdir(parents=True, exist_ok=True)
                file_path = data_dir / f"{uid}_{key}.flac"
                sf.write(file_path, val["value"], 16000, format="FLAC")
                line_parts.append(str(file_path))
            else:
                raise ValueError(
                    f"output type {val['type']} is not supported."
                    + "Please override the write function"
                )

            with open(scp_path, "a") as f:
                f.write(" ".join(line_parts) + "\n")

    def run_on_example(self, uid: str, sample: dict) -> dict:
        model = self.initialize_model()
        if self.stream and "audio_path" in sample:
            sample["stream"] = self.read("audio", sample["audio_path"], stream=True)
        model, sample = self.pre_inference(model, sample)
        with torch.no_grad():
            if self.stream and isinstance(sample.get("stream"), Iterator):
                outputs = []
                for chunk in sample["stream"]:
                    out = self.inference_body(model, chunk)
                    outputs.append(out)
                return self.post_inference(model, outputs)
            else:
                return self.inference_body(model, sample)

    def run_on_dataset(self, dataset_key: Any, output_dir: Union[str, Path]):
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        if self.parallel_config:
            self._run_parallel(dataset_key, output_dir)
        else:
            self._run_serial(dataset_key, output_dir)

    def _run_serial(self, dataset_key: Any, output_dir: Union[str, Path]):
        model = self._initialize_model(self.device)
        dataset = self._initialize_dataset(dataset_key)

        output_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        proc = psutil.Process(pid)
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 \
            if torch.cuda.is_available() else 0

        for idx in tqdm(range(len(dataset))):
            uid, sample = dataset[idx]
            if self.stream and "audio_path" in sample:
                sample["stream"] = self.read("audio", sample["audio_path"], stream=True)

            start_total = time.time()
            mem_before = proc.memory_info().rss / 1024 / 1024
            gpu_before = gpu_mem()

            t0 = time.time()
            model, sample = self.pre_inference(model, sample)
            pre_time = time.time() - t0

            outputs = []
            infer_times = []

            if self.stream and isinstance(sample.get("stream"), Iterator):
                for chunk in sample["stream"]:
                    t1 = time.time()
                    out = self.inference_body(model, chunk)
                    infer_times.append(time.time() - t1)
                    outputs.append(out)
            else:
                t1 = time.time()
                outputs = [self.inference_body(model, sample)]
                infer_times.append(time.time() - t1)

            t_post = time.time()
            result = self.post_inference(model, outputs)
            post_time = time.time() - t_post

            total_time = time.time() - start_total
            mem_after = proc.memory_info().rss / 1024 / 1024
            gpu_after = gpu_mem()

            result["pre_time"] = str(round(pre_time, 4))
            result["infer_time"] = " ".join(str(round(t, 4)) for t in infer_times)
            result["post_time"] = str(round(post_time, 4))
            result["total_time"] = str(round(total_time, 4))
            result["cpu_mem_MB"] = str(round(mem_after - mem_before, 2))
            result["gpu_mem_MiB"] = str(round(gpu_after - gpu_before, 2))

            self.write(uid, result, output_dir)

    def _run_parallel(self, dataset_key: Any, output_dir: Union[str, Path]):
        """Run inference in parallel using Dask"""
        set_parallel(self.parallel_config)
        output_dir.mkdir(parents=True, exist_ok=True)

        def process_sample(idx: int):
            """Process a single sample in a worker"""
            worker = get_worker()
            model = worker.model

            # Initialize dataset per worker
            dataset = worker.inference_runner.initialize_dataset(dataset_key)
            uid, sample = dataset[idx]

            try:
                # Track resources
                pid = os.getpid()
                proc = psutil.Process(pid)
                mem_before = proc.memory_info().rss / 1024 / 1024
                gpu_before = (
                    torch.cuda.memory_allocated() / 1024 / 1024
                    if torch.cuda.is_available()
                    else 0
                )

                t_total_start = time.time()

                # Prepare streaming if enabled
                if worker.inference_runner.stream and "audio_path" in sample:
                    sample["stream"] = worker.inference_runner.read(
                        "audio", sample["audio_path"], stream=True
                    )

                # Pre-inference
                t0 = time.time()
                model, sample = worker.inference_runner.pre_inference(model, sample)
                pre_time = time.time() - t0

                outputs = []
                infer_times = []

                # Run inference
                if worker.inference_runner.stream and isinstance(
                    sample.get("stream"), Iterator
                ):
                    for chunk in sample["stream"]:
                        t1 = time.time()
                        out = worker.inference_runner.inference_body(model, chunk)
                        infer_times.append(time.time() - t1)
                        outputs.append(out)
                else:
                    t1 = time.time()
                    result = worker.inference_runner.inference_body(model, sample)
                    infer_times.append(time.time() - t1)
                    outputs = [result]

                # Post-inference
                t2 = time.time()
                result = worker.inference_runner.post_inference(model, outputs)
                post_time = time.time() - t2

                # Ensure result is a dictionary
                if not isinstance(result, dict):
                    result = {"output": result}

                # Measure resource usage
                total_time = time.time() - t_total_start
                mem_after = proc.memory_info().rss / 1024 / 1024
                gpu_after = (
                    torch.cuda.memory_allocated() / 1024 / 1024
                    if torch.cuda.is_available()
                    else 0
                )

                # Add metrics to result
                result.update(
                    {
                        "pre_time": str(round(pre_time, 4)),
                        "infer_time": " ".join(str(round(t, 4)) for t in infer_times),
                        "post_time": str(round(post_time, 4)),
                        "total_time": str(round(total_time, 4)),
                        "cpu_mem_MB": str(round(mem_after - mem_before, 2)),
                        "gpu_mem_MiB": str(round(gpu_after - gpu_before, 2)),
                    }
                )

                return uid, result

            except Exception as e:
                error_msg = str(e)
                worker.inference_runner.on_error(uid, e)
                return uid, {"error": error_msg}

        # Create plugin with self reference
        plugin = InferencePlugin(self)

        # Get dataset length (temporary dataset just to get length)
        dummy_dataset = self.initialize_dataset(dataset_key)
        dataset_length = len(dummy_dataset)
        del dummy_dataset

        # Run with Dask client
        with get_client(plugin=plugin) as client:
            futures = client.map(process_sample, list(range(dataset_length)))
            for future in tqdm(as_completed(futures)):
                try:
                    uid, output = future.result()
                    if output is not None and "error" not in output:
                        self.write(uid, output, output_dir)
                    elif "error" in output:
                        logging.error(f"Error processing {uid}: {output['error']}")
                except Exception as e:
                    logging.error(f"Error in future: {e}")

    # ---- Hook methods to override ----
    def pre_inference(self, model, sample: dict):
        return model, sample

    @abstractmethod
    def inference_body(self, model, sample: Union[dict, Any]) -> dict:
        raise NotImplementedError

    def post_inference(self, model, outputs: list) -> dict:
        return outputs[0]

    # def on_error(self, uid: str, err: Exception):
    #     print(f"[ERROR] {uid}: {err}", flush=True)
