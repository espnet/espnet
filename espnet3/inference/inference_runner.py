import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import psutil
import soundfile as sf
import torch
from dask.distributed import WorkerPlugin, as_completed, get_worker
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from espnet2.train.preprocessor import AbsPreprocessor
from espnet3.data import (
    DatasetWithTransform,
    do_nothing_transform,
)
from espnet3.parallel import get_client, set_parallel

logging.basicConfig(level=logging.INFO)


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


def measure_resources(proc: psutil.Process):
    mem = proc.memory_info().rss / 1024 / 1024
    gpu = (
        torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    )
    return mem, gpu


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
        async_mode: bool = False,
    ):
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.stream = stream
        self.parallel_config = parallel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.dataset = None
        self.current_dataset_key = None
        self.async_mode = async_mode

    def initialize_model(self, device=None):
        """
        Instantiate and return the model using the model configuration.

        Users are expected to override this method if they need custom model
        initialization logic.

        Args:
            device (str, optional): The device to load the model on, e.g., 'cpu' or 'cuda'.

        Returns:
            Any: The instantiated model object using Hydra instantiation.
        """
        return instantiate(self.model_config)

    def _initialize_model(self, device=None):
        if device == "cuda" and self.parallel_config.env == "local_gpu":
            device = f"cuda:{os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]}"

        if self.model is None:
            self.model = self.initialize_model(device)

        return self.model

    def initialize_dataset(self, dataset_key, dataset_config=None):
        """
        Instantiate a test dataset and apply preprocessing and transformation.

        Users are expected to override this method if they need custom dataset
        initialization logic.

        Args:
            dataset_key (str): The name of the test dataset to initialize.
            dataset_config (DictConfig, optional): Configuration for all datasets.
                If None, uses `self.dataset_config`.

        Returns:
            DatasetWithTransform: Dataset object with transforms and optional preprocessor applied.

        Raises:
            RuntimeError: If the dataset key is not found in the config.
        """
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

        return DatasetWithTransform(
            instantiate(ds_conf.dataset),
            transform,
            preprocessor,
            use_espnet_preprocessor=is_espnet_preprocessor,
        )

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
        """
        Read input data from file (either audio or text), supporting offline and streaming modes.

        This function is used to abstract input loading logic depending on:
        - the type of input (`audio` or `text`)
        - whether streaming is enabled (`stream=True`)

        Args:
            input_type (str): Type of input to read. Must be either:
                - "audio": reads waveform from file
                - "text": reads plain text from file
            path (str): Path to the input file.
            stream (bool): If True, reads the input in chunks (streaming mode). Otherwise, reads the full file.
            chunk_sec (float): Duration (in seconds) of each audio chunk when streaming.
            chunk_chars (int): Number of characters per text chunk when streaming.

        Returns:
            Union[np.ndarray, str, Iterator[np.ndarray], Iterator[str]]:
                - If `stream=False`: returns the full waveform (`np.ndarray`) or full text (`str`).
                - If `stream=True`: returns an iterator over chunks:
                    - For audio: `Iterator[np.ndarray]` (each chunk is waveform array)
                    - For text: `Iterator[str]` (each chunk is a string)

        Raises:
            ValueError: If `input_type` is not one of {"audio", "text"}.

        Notes:
            - In streaming mode, `audio_path` or `text_path` must be present in the sample dictionary
            so that the file can be opened in chunks.
            - For streaming inference, this method is commonly used to initialize a generator
            and store it in `sample["stream"]`, which is then iterated chunk-by-chunk.

        Example:
            >>> # Offline audio
            >>> data = runner.read("audio", "audio.wav", stream=False)
            >>> print(data.shape)  # full waveform

            >>> # Streaming audio
            >>> stream = runner.read("audio", "audio.wav", stream=True, chunk_sec=0.1)
            >>> for chunk in stream:
            ...     print(chunk.shape)  # chunked waveform

            >>> # Offline text
            >>> text = runner.read("text", "input.txt", stream=False)
            >>> print(text)

            >>> # Streaming text
            >>> for chunk in runner.read("text", "input.txt", stream=True, chunk_chars=10):
            ...     print(chunk)
        """
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
        """
        Write the model output to Kaldi-style SCP files, and save audio/image data if present.

        This function handles multiple output types (`text`, `audio`, `image`) and
        appends entries to their corresponding `{key}.scp` files. If the output is of
        type `audio` or `image`, it also writes the actual data to disk in a
        structured directory under `output_dir/data/{key}/`.

        Args:
            uid (str): Unique identifier for the input sample. Will be used in file names and SCP entries.
            output (Dict[str, Any]): Output dictionary produced by the model.
                Each key in the dict corresponds to an output type, and must have:
                    - "type": one of {"text", "audio", "image"}
                    - "value": the actual value (e.g., string, numpy array)
            output_dir (str or Path): Base directory where outputs and .scp files will be written.

        Raises:
            ValueError: If an unsupported output type or output value structure is encountered.

        Notes:
            - Text outputs are directly written as string values into `{key}.scp`:
                Example line: `utt1 hello world`
            - Audio outputs (numpy arrays) are saved as FLAC files and referenced in `{key}.scp`:
                Output file: `data/audio/utt1_audio.flac`
            - Image outputs (numpy arrays) are saved as PNG files and referenced in `{key}.scp`:
                Output file: `data/weight/utt1_weight.png`
            - All .scp files are *appended* to if they already exist.
            - Output directory structure:
                output_dir/
                ├── text.scp
                ├── audio.scp
                ├── weight.scp
                └── data/
                    ├── audio/
                    │   └── utt1.flac
                    └── weight/
                        └── utt1.png

        Example:
            >>> output = {
            ...     "text": {"type": "text", "value": "hello world"},
            ...     "audio": {"type": "audio", "value": np.zeros(16000)},
            ...     "weight": {"type": "image", "value": np.random.rand(64, 64)}
            ... }
            >>> runner.write("utt1", output, "decode/")

            # Result:
            # - decode/text.scp  includes: "utt1 hello world"
            # - decode/audio.scp includes: "utt1 decode/data/audio/utt1.flac"
            # - decode/weight.scp includes: "utt1 decode/data/weight/utt1.png"
        """
        output_dir = Path(output_dir)
        for key, val in output.items():
            line_parts = [uid]
            if isinstance(val, str):
                line_parts.append(val)
            elif isinstance(val, dict):
                val_type = val.get("type")
                val_value = val.get("value")
                if val_type == "text":
                    line_parts.append(val_value)
                elif val_type in ("audio", "image"):
                    ext = "flac" if val_type == "audio" else "png"
                    data_dir = output_dir / "data" / key
                    data_dir.mkdir(parents=True, exist_ok=True)
                    file_path = data_dir / f"{uid}.{ext}"
                    if val_type == "audio":
                        sf.write(file_path, val_value, 16000, format="FLAC")
                    elif val_type == "image":
                        if np.issubdtype(val_value.dtype, np.floating):
                            val_value = np.clip(val_value, 0, 1)  # Normalize if needed
                            val_value = (val_value * 255).astype(np.uint8)
                        import matplotlib.pyplot as plt

                        plt.imsave(file_path, val_value, format="png")
                    line_parts.append(str(file_path))
                else:
                    raise ValueError(f"Unsupported output type: {val_type}")
            else:
                raise ValueError(f"Unsupported output value type: {type(val)}")

            with open(output_dir / f"{key}.scp", "a") as f:
                f.write(" ".join(line_parts) + "\n")

    def run_on_example(self, uid: str, sample: dict) -> dict:
        """
        Run inference on a single example, including optional streaming.

        Args:
            uid (str): Unique identifier for the sample.
            sample (dict): Sample input containing audio/text data or file paths.

        Returns:
            dict: Final model output after inference (including post-processing).
        """
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

    def _get_uid_sample(self, idx, example):
        if isinstance(example, tuple):
            uid, sample = example
        elif isinstance(example, dict):
            uid = str(idx)
            sample = example
        else:
            raise RuntimeError(f"Not supported type {type(example)}")
        return uid, sample

    def process_sample_core(
        self,
        sample: dict,
        model,
        read_fn,
        pre_fn,
        infer_fn,
        post_fn,
        stream: bool,
        proc: Optional[psutil.Process] = None,
    ) -> Dict[str, Any]:
        """
        Perform a complete inference pass on a single sample, including optional streaming,
        and record timing and memory usage metrics throughout the process.

        This method is responsible for executing the full inference pipeline on a single input sample:
            1. Streaming setup (if applicable)
            2. Preprocessing via `pre_fn`
            3. Model inference via `infer_fn` (single pass or streamed)
            4. Postprocessing via `post_fn`
            5. Resource usage and timing measurement

        Args:
            sample (dict): The input sample to process. It may include fields such as
                "audio_path", "text_path", or pre-loaded features depending on the model.
            model: The model instance used for inference.
            read_fn (Callable): Function to read input data, typically `self.read`.
            pre_fn (Callable): Preprocessing hook, typically `self.pre_inference`.
            infer_fn (Callable): Inference hook, typically `self.inference_body`.
            post_fn (Callable): Postprocessing hook, typically `self.post_inference`.
            stream (bool): If True, stream the input in chunks instead of full input.
            proc (psutil.Process, optional): A `psutil.Process` instance used to
                measure CPU memory and GPU memory before and after inference. If None,
                memory metrics will not be recorded.

        Returns:
            Dict[str, Any]: Output dictionary containing:
                - Model output from `post_fn`
                - Timing metrics:
                    - "pre_time": Time taken by preprocessing step (seconds)
                    - "infer_time": Space-separated string of inference durations for each chunk (seconds)
                    - "post_time": Time taken by postprocessing step (seconds)
                    - "total_time": Total elapsed time from start to end (seconds)
                - Resource usage metrics (if `proc` is provided):
                    - "cpu_mem_MB": Difference in resident memory usage (in megabytes)
                    - "gpu_mem_MiB": Difference in GPU memory allocation (in MiB, CUDA only)

        Timing Details:
            - `pre_time`: Time spent in `pre_fn` (usually lightweight normalization or tokenization)
            - `infer_time`: Per-chunk or single-pass inference time(s), may be multiple entries if streaming
            - `post_time`: Time spent in `post_fn` (e.g., merging chunk results, decoding)
            - `total_time`: Total wall-clock time including all the above stages

        Resource Measurement Details:
            - `cpu_mem_MB`: Difference in resident memory (RSS) before and after processing
            - `gpu_mem_MiB`: Difference in CUDA memory allocated before and after processing.
                Only recorded if CUDA is available and `torch.cuda.is_available()` returns True.

        Notes:
            - This method is designed to be framework-agnostic and easily parallelizable.
            - It is internally used by serial, parallel, and async decoding paths.
            - All returned numerical values are strings to make them easily writable to log files or .scp metadata.

        Example:
            >>> result = runner.process_sample_core(
            ...     sample,
            ...     model,
            ...     runner.read,
            ...     runner.pre_inference,
            ...     runner.inference_body,
            ...     runner.post_inference,
            ...     stream=True,
            ...     proc=psutil.Process(os.getpid())
            ... )
            >>> print(result["total_time"], result["cpu_mem_MB"])
        """
        if stream and "audio_path" in sample:
            sample["stream"] = read_fn("audio", sample["audio_path"], stream=True)

        mem_before = gpu_before = 0
        if proc:
            mem_before, gpu_before = measure_resources(proc)

        t_start = time.time()
        t0 = time.time()
        model, sample = pre_fn(model, sample)
        pre_time = time.time() - t0

        outputs, infer_times = [], []
        stream_data = sample.get("stream")

        if stream and isinstance(stream_data, Iterator):
            for chunk in stream_data:
                t1 = time.time()
                outputs.append(infer_fn(model, chunk))
                infer_times.append(time.time() - t1)
        else:
            t1 = time.time()
            outputs = [infer_fn(model, sample)]
            infer_times.append(time.time() - t1)

        t2 = time.time()
        result = post_fn(model, outputs)
        post_time = time.time() - t2
        total_time = time.time() - t_start

        if proc:
            mem_after, gpu_after = measure_resources(proc)
            result.update(
                {
                    "cpu_mem_MB": str(round(mem_after - mem_before, 2)),
                    "gpu_mem_MiB": str(round(gpu_after - gpu_before, 2)),
                }
            )

        result.update(
            {
                "pre_time": str(round(pre_time, 4)),
                "infer_time": " ".join(str(round(t, 4)) for t in infer_times),
                "post_time": str(round(post_time, 4)),
                "total_time": str(round(total_time, 4)),
            }
        )
        return result

    def run_on_dataset(
        self, dataset_key: Any, output_dir: Union[str, Path], async_decode: bool = False
    ):
        """
        Run inference on a single test dataset, either serially or in parallel.

        This method acts as the entry point for executing inference on one of the
        test sets defined in the dataset configuration. It automatically selects the
        appropriate execution mode based on the presence of a parallel configuration
        (`self.parallel_config`) and the `async_decode` flag.

        Args:
            dataset_key (str): The name of the test dataset (e.g., "test-clean", "test-other")
                as defined under the `test` field in the DataOrganizer configuration.
            output_dir (Union[str, Path]): Path to the output directory where results
                (e.g., text.scp, audio.scp, image.scp) will be written. Can be a string or Path.
            async_decode (bool, optional): If True, run decoding in asynchronous mode using Dask
                (i.e., batches assigned per worker). If False, decoding is either serial
                (no Dask) or synchronous parallel (each sample is a Dask task).
                Default: False

        Raises:
            RuntimeError: If dataset_key is not found or configuration is incomplete.
            Any exception raised during sample inference will be caught and logged.

        Execution Modes:
            - Serial:
                If `self.parallel_config` is None, samples are processed one-by-one on a single process.
                This mode is simple and deterministic, and suitable for debugging.

            - Parallel (sync):
                If `self.parallel_config` is provided and `async_decode` is False,
                each sample is submitted to Dask as an independent task. This allows
                finer-grained load balancing but may incur higher overhead for large datasets.

            - Parallel (async):
                If `self.parallel_config` is provided and `async_decode` is True,
                the dataset is divided into index chunks, each processed by a dedicated Dask worker.
                This mode is more efficient for large datasets and streaming inference.

        Integration with Dataset Organizer:
            This method assumes datasets are defined using `espnet3.data.DataOrganizer`,
            where each test set is registered with a name and instantiation config:

                dataset:
                  _target_: espnet3.data.DataOrganizer
                  test:
                    - name: test-clean
                      dataset:
                        _target_: ...
                    - name: test-other
                      dataset:
                        _target_: ...

            The `dataset_key` provided to this method must match one of those names
            (e.g., "test-clean").

        Example:
            >>> runner = InferenceRunner(
            ...     model_config=cfg.model,
            ...     dataset_config=cfg.dataset,
            ...     stream=True,
            ...     parallel=cfg.parallel,
            ... )
            >>> runner.run_on_dataset("test-clean", Path("decode/test-clean"))

        See Also:
            - `initialize_dataset()`: How datasets are resolved from name.
            - `write()`: How model outputs are written to .scp files.
            - `process_sample_core()`: Core streaming-aware inference logic.
        """
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        if self.parallel_config and async_decode:
            self._run_parallel_async(dataset_key, output_dir)
        elif self.parallel_config:
            self._run_parallel(dataset_key, output_dir)
        else:
            self._run_serial(dataset_key, output_dir)

    def _run_serial(self, dataset_key: Any, output_dir: Union[str, Path]):
        model = self._initialize_model(self.device)
        dataset = self._initialize_dataset(dataset_key)
        output_dir.mkdir(parents=True, exist_ok=True)
        proc = psutil.Process(os.getpid())

        for idx in tqdm(range(len(dataset))):
            uid, sample = self._get_uid_sample(idx, dataset[idx])
            try:
                result = self.process_sample_core(
                    sample,
                    model,
                    self.read,
                    self.pre_inference,
                    self.inference_body,
                    self.post_inference,
                    self.stream,
                    proc,
                )
                self.write(uid, result, output_dir)
            except Exception as e:
                logging.error(f"[{uid}] Error: {e}")

    def _run_parallel_async(self, dataset_key: Any, output_dir: Union[str, Path]):
        """Run inference in parallel using Dask"""
        # === 1. Setup Dask parallel environment ===
        self.parallel_config.options.asynchronous = True
        set_parallel(self.parallel_config)
        output_dir.mkdir(parents=True, exist_ok=True)

        # === 2. Determine dataset length and split indices ===
        dummy_dataset = self.initialize_dataset(dataset_key)
        dataset_length = len(dummy_dataset)
        del dummy_dataset

        n_workers = self.parallel_config.n_workers
        index_chunks = [
            chunk.tolist()
            for chunk in np.array_split(np.arange(dataset_length), n_workers)
        ]

        def process_samples(
            indices: List[int], dataset_key: str, stream: bool, base_output_dir: str
        ):
            worker = get_worker()
            model = worker.inference_runner.initialize_model()
            dataset = worker.inference_runner.initialize_dataset(dataset_key)

            short_id = worker.name.replace(
                "worker-", ""
            )  # or str(abs(hash(worker.id)) % 100000)
            sub_output_dir = Path(base_output_dir) / f"process-{short_id}"
            proc = psutil.Process(os.getpid())

            for idx in indices:
                example = dataset[idx]
                uid, sample = self._get_uid_sample(idx, example)
                try:
                    result = self.process_sample_core(
                        sample,
                        model,
                        self.read,
                        self.pre_inference,
                        self.inference_body,
                        self.post_inference,
                        self.stream,
                        proc,
                    )
                    self.write(uid, result, sub_output_dir)
                except Exception as e:
                    logging.error(f"[{uid}] Error: {e}")

            return None

        # === 3. Define plugin ===
        plugin = InferencePlugin(self)

        # === 4. Launch Dask submit jobs ===
        futures = []
        with get_client(plugin=plugin) as client:
            for indices in index_chunks:
                future = client.submit(
                    process_samples,  # assumed to be imported or defined elsewhere
                    indices,
                    dataset_key,
                    self.stream,
                    str(output_dir),
                )
                futures.append(future)

            # === 5. Wait for completion and error reporting ===
            for future in tqdm(futures, desc="Awaiting async jobs"):
                try:
                    future.result()  # writing is done inside each worker
                except Exception as e:
                    logging.error(f"[async job error] {e}")

        # === 6. Merge .scp.* files into final output ===
        merged_output_dir = output_dir
        self.merge_scp_files(output_dir, merged_output_dir)

    def _run_parallel(self, dataset_key: Any, output_dir: Union[str, Path]):
        self.parallel_config.options.asynchronous = False
        set_parallel(self.parallel_config)
        dataset = self.initialize_dataset(dataset_key)
        dataset_len = len(dataset)
        plugin = InferencePlugin(self)

        with get_client(plugin=plugin) as client:

            def worker_fn(idx: int):
                worker = get_worker()
                dataset = worker.inference_runner.initialize_dataset(dataset_key)
                uid, sample = worker.inference_runner._get_uid_sample(idx, dataset[idx])
                proc = psutil.Process(os.getpid())
                try:
                    result = self.process_sample_core(
                        sample,
                        worker.model,
                        worker.inference_runner.read,
                        worker.inference_runner.pre_inference,
                        worker.inference_runner.inference_body,
                        worker.inference_runner.post_inference,
                        worker.inference_runner.stream,
                        proc,
                    )
                    return uid, result
                except Exception as e:
                    return uid, {"error": str(e)}

            futures = client.map(worker_fn, list(range(dataset_len)))
            for fut in tqdm(as_completed(futures)):
                uid, result = fut.result()
                if "error" in result:
                    logging.error(f"[{uid}] Error: {result['error']}")
                else:
                    self.write(uid, result, output_dir)

    # ---- Hook methods to override ----
    def pre_inference(self, model, sample: dict):
        """
        Preprocess the sample before inference. Override if needed.

        Args:
            model: Model instance.
            sample (dict): Input sample.

        Returns:
            Tuple: (model, processed sample)
        """
        return model, sample

    @abstractmethod
    def inference_body(self, model, sample: Union[dict, Any]) -> dict:
        """
        Core inference logic to be implemented by subclasses.

        Args:
            model: Model instance.
            sample (dict or Any): Preprocessed sample or data chunk.

        Returns:
            dict: Inference output dictionary.

        Raises:
            NotImplementedError: Always, must be implemented by subclass.
        """
        raise NotImplementedError

    def post_inference(self, model, outputs: list) -> dict:
        """
        Post-process outputs after streaming inference.

        Args:
            model: Model instance.
            outputs (list): List of per-chunk or single inference outputs.

        Returns:
            dict: Final merged or selected output.
        """
        return outputs[0]

    def merge_scp_files(self, output_dir: Path, final_output_dir: Path, suffix=".scp"):
        """
        Merge multiple partial SCP files into unified output SCP files.

        Args:
            output_dir (Path): Directory containing partial .scp.* files.
            final_output_dir (Path): Destination for merged .scp files.
            suffix (str): File suffix to look for. Default is '.scp'.
        """
        final_output_dir.mkdir(parents=True, exist_ok=True)
        all_scp_files = list(
            output_dir.glob(f"*{suffix}.*")
        )  # e.g., text.scp.0, wav.scp.2

        scp_groups = {}
        for f in all_scp_files:
            key = f.name.split(".")[0] + suffix  # e.g., "text.scp"
            scp_groups.setdefault(key, []).append(f)

        for key, files in scp_groups.items():
            lines = []
            for f in sorted(files):
                lines.extend(f.read_text().splitlines())
            with open(final_output_dir / key, "w") as fout:
                fout.write("\n".join(lines) + "\n")
