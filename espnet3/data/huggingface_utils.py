import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple, Union

import datasets
import numpy as np
from dask.distributed import Client, LocalCluster, WorkerPlugin, get_worker
from espnetez.parallel import get_client, get_parallel_config, parallel_map
from lhotse.audio import Recording
from lhotse.audio.backend import (
    AudioBackend,
    FileObject,
    LibsndfileCompatibleAudioInfo,
    set_current_audio_backend,
)
from lhotse.audio.source import AudioSource, PathOrFilelike
from lhotse.cut import Cut, CutSet, MonoCut
from lhotse.supervision import SupervisionSegment
from lhotse.utils import Pathlike, Seconds
from omegaconf import DictConfig
from pathos.multiprocessing import ProcessingPool as Pool


class HuggingFaceAudioSource(AudioSource):
    type: str
    """
    The type of audio source. Supported types are:
    - 'file' (supports most standard audio encodings, possibly multi-channel)
    - 'command' [unix pipe] (supports most standard audio encodings, possibly multi-channel)
    - 'url' (any URL type that is supported by "smart_open" library, e.g. http/https/s3/gcp/azure/etc.)
    - 'memory' (any format, read from a binary string attached to 'source' member of AudioSource)
    - 'huggingface' (supports loading audio from HuggingFace datasets)
    - 'shar' (indicates a placeholder that will be filled later when using Lhotse Shar data format)
    """

    def _prepare_for_reading(
        self, offset: Seconds, duration: Optional[Seconds]
    ) -> PathOrFilelike:
        if self.type == "huggingface":
            return self.source
        else:
            return super()._prepare_for_reading(offset, duration)


def is_datasets_available() -> bool:
    try:
        import datasets

        return True
    except ImportError:
        return False


class HuggingfaceDatasetsBackend(AudioBackend):
    """
    A backend for reading audio data from HuggingFace datasets.
    This class supports audio data with #channels=1.
    If you want to read audio data with more channels, please copy this class
    and modify the `read_audio` method to support more channels.
    """

    def __init__(self, dataset_id: str = None):
        os.environ["LHOTSE_AUDIO_BACKEND"] = "HuggingfaceDatasetsBackend"
        try:
            worker = get_worker()
            self.dataset = worker.dataset
            self.dataset_id = worker.dataset_id
        except:
            assert dataset_id is not None, "Dataset id is not provided"
            self.dataset = _load_from_hf_or_disk(dataset_id)
            self.dataset_id = dataset_id

    def read_audio(
        self,
        path_or_fd: str,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
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
        return is_datasets_available()

    def supports_save(self) -> bool:
        return False

    def supports_info(self) -> bool:
        return True

    def info(
        self,
        path_or_fd: str,
        force_opus_sampling_rate: Optional[int] = None,
    ):
        data = self.dataset[int(path_or_fd)]
        return LibsndfileCompatibleAudioInfo(
            channels=1,
            frames=len(data["audio"]["array"]),
            samplerate=data["audio"]["sampling_rate"],
            duration=len(data["audio"]["array"]) / data["audio"]["sampling_rate"],
        )


def _load_from_hf_or_disk(path_or_hf):
    try:
        return datasets.load_dataset(path_or_hf)
    except:
        return datasets.load_from_disk(path_or_hf)


class HuggingfaceAudioLoader(WorkerPlugin):
    def __init__(self, dataset_id, split: str = None) -> None:
        super().__init__()
        self.dataset_id = dataset_id
        self.split = split

    def setup(self, worker):
        # worker.dataset = self.load_dataset()
        worker.dataset = _load_from_hf_or_disk(self.dataset_id)
        worker.dataset_id = self.dataset_id
        worker.split = self.split

    def load_dataset(self):
        try:
            return datasets.load_dataset(self.dataset_id)
        except:
            return datasets.load_from_disk(self.dataset_id)


def cut_from_huggingface(idx: int, data_info: Dict) -> Cut:
    """
    Create a Cut from a single example from a HuggingFace dataset.
    If data_info is provided, it should contain the mappings from
    huggingface columns to lhotse supervision keys.
    This function is implemented mainly because we want to procell in parallel.

    Args:
        example: A single example from a HuggingFace dataset.
        data_info: A dictionary containing the mappings from

    Returns:
        A Cut object.
    """
    # create a recording.
    worker = get_worker()
    ds = worker.dataset[worker.split]
    data_id = str(idx)
    data = ds[idx]
    duration = len(data["audio"]["array"]) / data["audio"]["sampling_rate"]
    sources = [
        HuggingFaceAudioSource(
            type="huggingface",
            source=f"{worker.dataset_id}:{worker.split}:{data_id}",
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
    Convert a HuggingFace dataset into a CutSet using Dask-based parallel processing.
    We need dataset_id and split to specify the audio source.

    Args:
        dataset: HuggingFace dataset
        data_info (Dict[str, callable]): Mapping from HuggingFace columns to supervision fields
        client (Optional[Union[Client, DictConfig]]): Either a Dask client or a parallel config dict.
            If not provided, it assumes `parallel.set()` has already been called.
        dataset_id: HuggingFace dataset ID
        split: HuggingFace split to use (default: None, uses the 'train' split)


    Returns:
        CutSet: A combined CutSet from all entries

    Raises:
        RuntimeError: If no client is set and no config is provided.

    Example:
        >>> cuts = cutset_from_huggingface(ds, info, client=config)  # config is DictConfig
        >>> cuts = cutset_from_huggingface(ds, info)  # requires parallel.set() to be called first
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
        try:
            with get_client(get_parallel_config()) as client:
                if not isinstance(client, LocalCluster):
                    client.register_worker_plugin(
                        worker_plugin
                    )  # register worker plugin for efficient data transfer.
                cuts = parallel_map(
                    runner, list(range(dataset_length)), client=client
                )  # uses global client
        except RuntimeError:
            raise RuntimeError(
                "No Dask client available. Please call parallel.set(config) or pass a config/client explicitly."
            )
    else:
        raise TypeError("client must be None, a Dask Client, or a DictConfig.")

    return CutSet.from_cuts(cuts)
