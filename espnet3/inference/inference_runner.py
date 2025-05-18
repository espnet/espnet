import psutil
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union
from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf
import torch
from dask.distributed import WorkerPlugin, as_completed, get_worker
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from espnet3 import get_espnet_model
from espnet3.data import DataOrganizer
from espnet3.parallel import get_client, set_parallel

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
        stream: bool = False,
        parallel_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = None
        self.stream = stream
        self.parallel_config = parallel_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_model(self):
        if self.model is None:
            pass  # subclass handles actual model creation

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
            if val["type"] == "text":
                line_parts.append(val["value"])
            elif val["type"] == "audio":
                data_dir = output_dir / "data" / key
                data_dir.mkdir(parents=True, exist_ok=True)
                file_path = data_dir / f"{uid}_{key}.flac"
                sf.write(file_path, val["value"], 16000, format="FLAC")
                line_parts.append(str(file_path))
            else:
                continue
            with open(scp_path, "a") as f:
                f.write(" ".join(line_parts) + "\n")

    def run_on_example(self, uid: str, sample: dict) -> dict:
        self.initialize_model()
        if self.stream and "audio_path" in sample:
            sample["stream"] = self.read("audio", sample["audio_path"], stream=True)
        self.initialize(sample)
        self.pre_inference(sample)
        with torch.no_grad():
            if self.stream and isinstance(sample.get("stream"), Iterator):
                outputs = []
                for chunk in sample["stream"]:
                    out = self.inference_body(chunk)
                    outputs.append(out)
                return self.post_inference(outputs)
            else:
                return self.inference_body(sample)

    def run_on_dataset(self, dataset: Any, output_dir: Path):
        if self.parallel_config:
            self._run_parallel(dataset, output_dir)
        else:
            self._run_serial(dataset, output_dir)

    def _run_serial(self, dataset: Any, output_dir: Path):
        self.initialize_model()
        output_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        proc = psutil.Process(pid)
        gpu_mem = lambda: torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

        logs = {k: {} for k in ["pre_time", "infer_time", "post_time", "total_time", "cpu_mem_MB", "gpu_mem_MiB"]}

        for idx in tqdm(range(len(dataset))):
            uid, sample = dataset[idx]
            try:
                if self.stream and "audio_path" in sample:
                    sample["stream"] = self.read("audio", sample["audio_path"], stream=True)

                start_total = time.time()
                mem_before = proc.memory_info().rss / 1024 / 1024
                gpu_before = gpu_mem()

                self.initialize(sample)

                t0 = time.time()
                self.pre_inference(sample)
                pre_time = time.time() - t0

                outputs = []
                infer_times = []

                if self.stream and isinstance(sample.get("stream"), Iterator):
                    for chunk in sample["stream"]:
                        t1 = time.time()
                        out = self.inference_body(chunk)
                        infer_times.append(time.time() - t1)
                        outputs.append(out)
                    t_post = time.time()
                    result = self.post_inference(outputs)
                    post_time = time.time() - t_post
                else:
                    t1 = time.time()
                    result = self.inference_body(sample)
                    infer_times.append(time.time() - t1)
                    post_time = 0

                total_time = time.time() - start_total
                mem_after = proc.memory_info().rss / 1024 / 1024
                gpu_after = gpu_mem()

                self.write(uid, result, output_dir)

                logs["pre_time"][uid] = str(round(pre_time, 4))
                logs["infer_time"][uid] = " ".join(str(round(t, 4)) for t in infer_times)
                logs["post_time"][uid] = str(round(post_time, 4))
                logs["total_time"][uid] = str(round(total_time, 4))
                logs["cpu_mem_MB"][uid] = str(round(mem_after - mem_before, 2))
                logs["gpu_mem_MiB"][uid] = str(round(gpu_after - gpu_before, 2))

            except Exception as e:
                self.on_error(uid, e)

        for key, content in logs.items():
            with open(output_dir / f"{key}.scp", "w") as f:
                for uid, val in content.items():
                    f.write(f"{uid} {val}\n")

    def _run_parallel(self, dataset: Any, output_dir: Path):
        class InferencePlugin(WorkerPlugin):
            def __init__(self, model_config: Dict[str, Any], dataset: Any):
                self.model_config = model_config
                self.dataset = dataset

            def setup(self, worker):
                model = self.initialize_model()
                worker.model = model.to(worker.device)
                worker.dataset = self.dataset
                worker.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_parallel(self.parallel_config)
        output_dir.mkdir(parents=True, exist_ok=True)

        def process(idx: int):
            import os, time, torch, psutil
            from pathlib import Path

            model = get_worker().model
            dataset = get_worker().dataset
            uid, sample = dataset[idx]
            result = {
                "uid": uid,
                "error": None,
                "output": None,
                "timing": {},
            }
            try:
                pid = os.getpid()
                proc = psutil.Process(pid)
                mem_before = proc.memory_info().rss / 1024 / 1024
                gpu_before = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

                t_total_start = time.time()

                t0 = time.time()
                pre = model.pre_inference(sample)
                pre_time = time.time() - t0

                t1 = time.time()
                out = model.inference_body(sample)
                infer_time = time.time() - t1

                t2 = time.time()
                post = model.post_inference([out])
                post_time = time.time() - t2

                total_time = time.time() - t_total_start
                mem_after = proc.memory_info().rss / 1024 / 1024
                gpu_after = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

                result["output"] = post
                result["timing"] = {
                    "pre_time": round(pre_time, 4),
                    "infer_time": round(infer_time, 4),
                    "post_time": round(post_time, 4),
                    "total_time": round(total_time, 4),
                    "cpu_mem_MB": round(mem_after - mem_before, 2),
                    "gpu_mem_MiB": round(gpu_after - gpu_before, 2),
                }

            except Exception as e:
                result["error"] = str(e)
            return result

        with get_client(plugin=InferencePlugin(self.model_config, dataset)) as client:
            futures = client.map(process, list(range(len(dataset))))
            for future in tqdm(as_completed(futures)):
                uid, output, err = future.result()
                if err is not None:
                    self.on_error(uid, Exception(err))
                elif output is not None:
                    self.write(uid, output, output_dir)

    # ---- Hook methods to override ----
    def initialize(self, sample: dict): pass

    def pre_inference(self, sample: dict): pass

    @abstractmethod
    def inference_body(self, chunk: Union[dict, Any]) -> dict:
        raise NotImplementedError
    
    def post_inference(self, outputs: list) -> dict:
        return outputs[-1] if outputs else {}
    
    def on_error(self, uid: str, err: Exception):
        print(f"[ERROR] {uid}: {err}", flush=True)

