import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple, Union, Any
from omegaconf import DictConfig, ListConfig, OmegaConf

import datasets
from datasets import load_dataset  # HuggingFace Datasets
import numpy as np
import torch
from dask.distributed import Client, LocalCluster, WorkerPlugin, get_worker
from espnet3.parallel import get_client, get_parallel_config, parallel_map
from lhotse.audio import Recording
from lhotse.audio.backend import (
    AudioBackend,
    FileObject,
    LibsndfileCompatibleAudioInfo,
    set_current_audio_backend,
    get_current_audio_backend,
)
from lhotse.audio.source import AudioSource, PathOrFilelike
from lhotse.cut import Cut, CutSet, MonoCut
from lhotse.supervision import SupervisionSegment
from lhotse.utils import Seconds, Pathlike
from lhotse.cut import CutSet, Cut, MonoCut
from dask.distributed import Client, get_worker, WorkerPlugin, LocalCluster
from dask.distributed import get_worker
from distributed.worker import thread_state

from espnet3.parallel import parallel_map, get_client, get_parallel_config



class HuggingFaceAudioSource(AudioSource):
    type: str
    """
    AudioSource subclass for HuggingFace datasets.

    This class overrides the method used to prepare audio for reading,
    allowing Lhotse to interface directly with audio samples embedded in
    HuggingFace datasets.

    Attributes:
        type (str): Audio source type. Must be 'huggingface' to activate special handling.
    """

    def _prepare_for_reading(
        self, offset: Seconds, duration: Optional[Seconds]
    ) -> PathOrFilelike:
        """
        Return the source path or handle for reading the audio.

        Args:
            offset (Seconds): Start offset in seconds.
            duration (Optional[Seconds]): Duration in seconds.

        Returns:
            PathOrFilelike: String identifier (for 'huggingface') or standard audio source.

        """
        if self.type == 'huggingface':
            return self.source
        else:
            return super()._prepare_for_reading(offset, duration)


def is_datasets_available() -> bool:
    """
    Check whether the HuggingFace 'datasets' library is available.

    Returns:
        bool: True if available, False otherwise.
    """
    try:
        import datasets

        return True
    except ImportError:
        return False


class HuggingfaceDatasetsBackend(AudioBackend):
    """
    Lhotse audio backend to load audio from HuggingFace datasets.

    Supports only mono-channel audio by default. Modify `read_audio` to support
    multi-channel if needed.

    Args:
        dataset_id (str, optional): Dataset ID to load, required outside Dask workers.
    """

    def __init__(self, dataset_id: str = None):
        os.environ["LHOTSE_AUDIO_BACKEND"] = "HuggingfaceDatasetsBackend"

        if self._running_in_dask_worker():
            worker = get_worker()
            assert hasattr(worker, "dataset"), "Worker has no attribute 'dataset'"
            assert hasattr(worker, "dataset_id"), "Worker has no attribute 'dataset_id'"
            self.dataset = worker.dataset
            self.dataset_id = worker.dataset_id
        else:
            assert dataset_id is not None, "Dataset id is not provided"
            self.dataset = _load_from_hf_or_disk(dataset_id)
            self.dataset_id = dataset_id
    
    def _running_in_dask_worker(self) -> bool:
        """
        Return True if we're running inside a Dask worker thread.
        """
        try:
            worker = get_worker()
            return True
        except:
            return False

    def read_audio(
        self,
        path_or_fd: str,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Read audio from a HuggingFace dataset entry.

        Args:
            path_or_fd (str): String in format 'dataset_id:split:index'.
            offset (Seconds, optional): Start offset in seconds. Default: 0.0.
            duration (Optional[Seconds]): Length to read in seconds. Default: None.
            force_opus_sampling_rate (Optional[int]): Not supported; raises error if set.

        Returns:
            Tuple[np.ndarray, int]: Audio waveform and sample rate.

        Raises:
            RuntimeError: If force_opus_sampling_rate is set.
            AssertionError: If dataset_id does not match.
        """
        # Check dataset id and split is correct
        dataset_id = path_or_fd.split(":")[0]
        split = path_or_fd.split(":")[1]
        data_idx = int(path_or_fd.split(":")[2])

        assert dataset_id == self.dataset_id, "Dataset id does not match"

        data = self.dataset[split][data_idx]
        if (
            force_opus_sampling_rate is not None
            and data["audio"]["sampling_rate"] != force_opus_sampling_rate
        ):
            raise RuntimeError(
                "force_opus_sampling_rate not available for "
                "HuggingfaceDatasetsBackend. Please modify the sampling_rate "
                "by using datasets' features."
            )

        audio = data["audio"]["array"]
        sampling_rate = data["audio"]["sampling_rate"]
        if offset is not None:
            start = int(offset * sampling_rate)
            audio = audio[start:]
        if duration is not None:
            end = int(duration * sampling_rate)
            audio = audio[:end]

        return audio, sampling_rate

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        """
        Check whether this backend is usable.

        Returns:
            bool: True if HuggingFace datasets are available.
        """
        return is_datasets_available()

    def supports_save(self) -> bool:
        """Returns False — saving not supported."""
        return False

    def supports_info(self) -> bool:
        """Returns True — info fetching is supported."""
        return True

    def info(
        self,
        path_or_fd: str,
        force_opus_sampling_rate: Optional[int] = None,
    ):
        """
        Return metadata for the audio in HuggingFace dataset.

        Args:
            path_or_fd (str): String identifier for the audio.
            force_opus_sampling_rate (Optional[int]): Ignored.

        Returns:
            LibsndfileCompatibleAudioInfo: Audio metadata.
        """
        # Check dataset id and split is correct
        dataset_id = path_or_fd.split(':')[0]
        split = path_or_fd.split(':')[1]
        data_idx = int(path_or_fd.split(':')[2])

        assert dataset_id == self.dataset_id, "Dataset id does not match"

        data = self.dataset[split][data_idx]
        return LibsndfileCompatibleAudioInfo(
            channels=1,
            frames=len(data["audio"]["array"]),
            samplerate=data["audio"]["sampling_rate"],
            duration=len(data["audio"]["array"]) / data["audio"]["sampling_rate"],
        )


def _load_from_hf_or_disk(path_or_hf, *args, **kwargs):
    """
    Attempt to load dataset from HuggingFace hub or local disk.

    Args:
        path_or_hf (str): Dataset path or identifier.

    Returns:
        datasets.DatasetDict: Loaded dataset.
    """
    if os.path.exists(path_or_hf):
        return datasets.load_from_disk(path_or_hf, *args, **kwargs)
    else:
        return datasets.load_dataset(path_or_hf, *args, **kwargs)


class HuggingfaceAudioLoader(WorkerPlugin):
    """
    Worker plugin to preload HuggingFace datasets into Dask workers.

    Args:
        dataset_id (str): Identifier of the dataset.
        split (str, optional): Dataset split to load.
    """
    def __init__(self, dataset_id, split: str = None) -> None:
        super().__init__()
        self.dataset_id = dataset_id
        self.split = split

    def setup(self, worker):
        """
        Setup hook that is run on each worker.

        Args:
            worker: Dask worker instance.
        """
        # worker.dataset = self.load_dataset()
        worker.dataset = _load_from_hf_or_disk(self.dataset_id)
        worker.dataset_id = self.dataset_id
        worker.split = self.split

    def load_dataset(self):
        """
        Load the dataset (legacy method, unused).

        Returns:
            datasets.DatasetDict
        """
        try:
            return datasets.load_dataset(self.dataset_id)
        except:
            return datasets.load_from_disk(self.dataset_id)


def cut_from_huggingface(idx: int, data_info: Dict) -> Cut:
    """
    Construct a MonoCut from a HuggingFace dataset example.

    Args:
        idx (int): Index of the example in the dataset.
        data_info (Dict): Mapping from supervision field -> function(example) -> value.

    Returns:
        Cut: A MonoCut with attached supervision.

    Example:
        >>> cut = cut_from_huggingface(0, {"text": lambda ex: ex["transcription"]})
    """
    # create a recording.
    data_id = str(idx)
    try:
        worker = get_worker()
        ds = worker.dataset[worker.split]
        data = ds[idx]
        dataset_id = worker.dataset_id
        split = worker.split
    except:
        backend = get_current_audio_backend()
        dataset = backend.dataset


    
    duration = len(data['audio']['array']) / data['audio']['sampling_rate']
    sources = [
        HuggingFaceAudioSource(
            type="huggingface",
            source=f"{dataset_id}:{split}:{data_id}",
            channels=[0],
        ),
    ]
    recording = Recording(
        id=data_id,
        sources=sources,
        sampling_rate=data["audio"]["sampling_rate"],
        num_samples=len(data["audio"]["array"]),
        duration=duration,
        channel_ids=[0],
    )

    # create a supervision.
    supervision_dict = dict(
        id=data_id,
        recording_id=data_id,
        start=0.0,
        duration=duration,
        channel=0,
    )
    for k, v in data_info.items():
        supervision_dict[k] = v(ds[idx])

    supervision = SupervisionSegment.from_dict(supervision_dict)
    cut = MonoCut(
        id=data_id,
        start=0.0,
        duration=duration,
        channel=0,
        supervisions=[supervision],
        recording=recording,
    )
    return cut


def cutset_from_huggingface(
    data_info: Dict[str, callable],
    dataset_length: int,
    dataset_id: str,
    split: str = None,
    client: Optional[Union[Client, DictConfig]] = None,
) -> CutSet:
    """
    Convert a HuggingFace dataset to a CutSet using parallel processing with Dask.

    Args:
        data_info (Dict[str, callable]): Mapping from supervision field name to function(example).
        dataset_length (int): Total number of dataset entries.
        dataset_id (str): HuggingFace dataset name or path.
        split (str, optional): Dataset split to use (e.g., "train").
        client (Optional[Union[Client, DictConfig]]): Dask client or parallel config.
            If not provided, global parallel settings are used.

    Returns:
        CutSet: The resulting CutSet with recordings and supervisions.

    Raises:
        RuntimeError: If no client is set and none can be created.

    Example:
        >>> cutset = cutset_from_huggingface(
                data_info={"text": lambda ex: ex["transcription"]},
                dataset_length=100,
                dataset_id="<huggingface_dataset_tag>",
                split="train",
                client=config.parallel
            )
    """
    worker_plugin = HuggingfaceAudioLoader(dataset_id, split)
    runner = partial(cut_from_huggingface, data_info=data_info)
    if isinstance(client, DictConfig):
        with get_client(client) as c:
            if not isinstance(client, LocalCluster):
                c.register_worker_plugin(worker_plugin)
            cuts = parallel_map(runner, list(range(dataset_length)), client=c)
    elif isinstance(client, Client):
        if not isinstance(client, LocalCluster):
            client.register_worker_plugin(
                worker_plugin
            )  # register worker plugin for efficient data transfer.
        cuts = parallel_map(runner, list(range(dataset_length)), client=client)
    elif client is None:
        parallel_config = get_parallel_config()
        if parallel_config is not None:
            with get_client(parallel_config) as client:
                if not isinstance(client, LocalCluster):
                    client.register_worker_plugin(worker_plugin)  # register worker plugin for efficient data transfer.
                cuts = parallel_map(runner, list(range(dataset_length)), client=client)  # uses global client
        else:
            cuts = [runner(idx) for idx in range(dataset_length)]
    else:
        raise TypeError("client must be None, a Dask Client, or a DictConfig.")

    return CutSet.from_cuts(cuts)


class HFWrapper:
    """
    Wrapper for Hugging Face datasets compatible with DataOrganizer interface.

    Attributes:
        path (str): HuggingFace dataset ID or local path.
        name (Optional[str]): Optional dataset config name.
        split (str): Split to load (e.g., 'train', 'validation').
        cache_dir (Optional[str]): Cache directory path.
        kwargs (Dict[str, Any]): Additional keyword arguments passed to load_dataset.
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # Check if there is a nested list or dict in the arguments from OmegaConf
        for k,v in kwargs.items():
            if isinstance(v, (ListConfig, DictConfig)):
                kwargs[k] = OmegaConf.to_container(v, resolve=True)

        self.dataset = load_dataset(
            *args,
            **kwargs,
        )
    
    def _convert_to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.numpy()
        elif isinstance(data, (list, tuple)):
            return [self._convert_to_numpy(x) for x in data]
        elif isinstance(data, dict):
            return {k: self._convert_to_numpy(v) for k, v in data.items()}
        else:
            return data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return (str(idx), self.dataset[idx])

    def __len__(self) -> int:
        return len(self.dataset)

    def __call__(self, idx: int) -> Dict[str, Any]:
        return self.__getitem__(idx)
