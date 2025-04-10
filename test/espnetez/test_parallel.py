from unittest import mock

import numpy as np
import pytest
from datasets import load_from_disk
from lhotse import CutSet, SupervisionSegment
from lhotse.audio.backend import LibsndfileCompatibleAudioInfo

from espnet3.data import (
    HuggingfaceAudioLoader,
    HuggingFaceAudioSource,
    HuggingfaceDatasetsBackend,
    cut_from_huggingface,
    cutset_from_huggingface,
)

DUMMY_DATASET_PATH = "test_utils/espnet3_dummy/espnet3_test_dataset"


def test_read_audio_basic():
    dataset = load_from_disk(DUMMY_DATASET_PATH)

    with mock.patch("dask.distributed.get_worker") as mock_worker:
        mock_worker.return_value.dataset = dataset
        mock_worker.return_value.dataset_id = DUMMY_DATASET_PATH

        backend = HuggingfaceDatasetsBackend(
            dataset_id=DUMMY_DATASET_PATH,
        )
        waveform, sr = backend.read_audio(f"{DUMMY_DATASET_PATH}:train:0")
        assert waveform.shape[0] == 16000
        assert sr == 16000


def test_read_audio_with_offset_and_duration():
    dataset = load_from_disk(DUMMY_DATASET_PATH)

    with mock.patch("dask.distributed.get_worker") as mock_worker:
        mock_worker.return_value.dataset = dataset
        mock_worker.return_value.dataset_id = DUMMY_DATASET_PATH

        backend = HuggingfaceDatasetsBackend(dataset_id=DUMMY_DATASET_PATH)
        waveform, sr = backend.read_audio(
            f"{DUMMY_DATASET_PATH}:train:0", offset=0.5, duration=0.25
        )
        assert waveform.shape[0] == int(0.25 * sr)


# def test_cut_from_huggingface():
#     dataset = load_from_disk(DUMMY_DATASET_PATH)

#     with mock.patch("dask.distributed.get_worker") as mock_worker:
#         mock_worker.return_value.dataset = dataset
#         mock_worker.return_value.dataset_id = DUMMY_DATASET_PATH
#         mock_worker.return_value.split = "train"

#         cut = cut_from_huggingface(
#             idx=0,
#             data_info={"text": lambda ex: ex["transcription"]}
#         )

#         assert cut.recording.sampling_rate == 16000
#         assert cut.supervisions[0].text == "hello world"


# def test_cutset_from_huggingface():
#     with mock.patch("espnet3.parallel.get_client"), \
#          mock.patch("espnet3.parallel.parallel_map") as mock_pm, \
#          mock.patch("espnet3.data.HuggingfaceAudioLoader"):

#         dummy_cut = mock.MagicMock()
#         mock_pm.return_value = [dummy_cut, dummy_cut]

#         data_info = {"text": lambda ex: ex["transcription"]}
#         cuts = cutset_from_huggingface(
#             data_info=data_info,
#             dataset_length=2,
#             dataset_id=DUMMY_DATASET_PATH,
#             split="train",
#             client=None
#         )

#         assert isinstance(cuts, CutSet)
#         assert len(list(cuts.cuts.values())) == 2


def test_force_opus_sampling_rate_raises():
    dataset = load_from_disk(DUMMY_DATASET_PATH)

    with mock.patch("dask.distributed.get_worker") as mock_worker:
        mock_worker.return_value.dataset = dataset
        mock_worker.return_value.dataset_id = DUMMY_DATASET_PATH

        backend = HuggingfaceDatasetsBackend(dataset_id=DUMMY_DATASET_PATH)
        with pytest.raises(RuntimeError):
            backend.read_audio(
                f"{DUMMY_DATASET_PATH}:train:0", force_opus_sampling_rate=8000
            )


def test_huggingface_audio_source_prepare_for_reading_type_hf():
    src = HuggingFaceAudioSource(
        type="huggingface", source=f"{DUMMY_DATASET_PATH}:train:0", channels=[0]
    )
    result = src._prepare_for_reading(offset=0.0, duration=1.0)
    assert result == f"{DUMMY_DATASET_PATH}:train:0"


def test_huggingface_audio_source_prepare_for_reading_fallback():
    with mock.patch(
        "lhotse.audio.source.AudioSource._prepare_for_reading"
    ) as mock_super:
        mock_super.return_value = "mocked_output"
        src = HuggingFaceAudioSource(
            type="file", source="test_utils/ctc_align_test.wav", channels=[0]
        )
        result = src._prepare_for_reading(offset=0.5, duration=1.0)
        mock_super.assert_called_once()
        assert result == "mocked_output"


def test_huggingface_datasets_backend_info():
    dataset = load_from_disk(DUMMY_DATASET_PATH)

    with mock.patch("dask.distributed.get_worker") as mock_worker:
        mock_worker.return_value.dataset = dataset
        mock_worker.return_value.dataset_id = DUMMY_DATASET_PATH

        backend = HuggingfaceDatasetsBackend(dataset_id=DUMMY_DATASET_PATH)
        info = backend.info(f"{DUMMY_DATASET_PATH}:train:0")

        assert isinstance(info, LibsndfileCompatibleAudioInfo)
        assert info.channels == 1
        assert info.samplerate == 16000
        assert info.frames == 16000
        assert abs(info.duration - 1.0) < 1e-3
