# Data preparation
import os

from datasets import load_dataset
from espnetez.data.lhotse_utils import (
    HuggingFaceAudioSource,
    HuggingfaceDatasetsBackend,
    cutset_from_huggingface,
)
from lhotse import CutSet, Fbank, Mfcc
from lhotse.audio import Recording
from lhotse.audio.backend import set_current_audio_backend
from lhotse.cut import MonoCut
from lhotse.supervision import SupervisionSegment

ds = load_dataset("saeedq/librispeech_100h")

set_current_audio_backend(HuggingfaceDatasetsBackend("saeedq/librispeech_100h"))


data_info = {
    "text": lambda x: x["text"],
    "language": lambda x: "English",
    "speaker": lambda x: x["audio"]["path"].split("-")[0],
    "channel": lambda x: 0,
}


from espnetez.data.lhotse_utils import HuggingfaceAudioLoader

## parallel
from espnetez.parallel import (
    get_client,
    get_parallel_config,
    parallel_map,
    set_parallel,
)
from omegaconf import OmegaConf

conf = OmegaConf.load("test.yaml")

set_parallel(conf.parallel)


cutset = cutset_from_huggingface(
    data_info, len(ds["test"]), "saeedq/librispeech_100h", "test"
)


os.environ["LHOTSE_AUDIO_BACKEND"] = "HuggingfaceDatasetsBackend"
# worker_plugin = HuggingfaceAudioLoader("saeedq/librispeech_100h", "test")


# with get_client(get_parallel_config()) as client:
#     client.register_worker_plugin(worker_plugin)
#     cuts_train = cutset.compute_and_store_features(Fbank(), "./tmp",num_jobs=16, executor=client)


# cuts_train = cutset.compute_and_store_features(Fbank(), "./tmp_1")

from lhotse.dataset.sampling import SimpleCutSampler
from lhotse.dataset.speech_recognition import K2SpeechRecognitionDataset

dataset = K2SpeechRecognitionDataset()

sampler = SimpleCutSampler(
    cuts=cutset,
    max_cuts=4,
    shuffle=True,
)
