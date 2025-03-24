
# Data preparation
from lhotse.supervision import SupervisionSegment
from lhotse.audio import Recording
from espnetez.data.lhotse_utils import HuggingFaceAudioSource, HuggingfaceDatasetsBackend, cutset_from_huggingface
from datasets import load_dataset
from lhotse.audio.backend import set_current_audio_backend
from lhotse.cut import MonoCut
from lhotse import CutSet, Mfcc, Fbank
import os


ds = load_dataset("saeedq/librispeech_100h")

set_current_audio_backend(HuggingfaceDatasetsBackend(
    "saeedq/librispeech_100h"
))


data_info = {
    "text": lambda x: x["text"],
    "language": lambda x: "English",
    "speaker": lambda x: x['audio']['path'].split("-")[0],
    "channel": lambda x: 0,
}



## parallel
from espnetez.parallel import parallel_map, set_parallel, get_client, get_parallel_config
from espnetez.data.lhotse_utils import HuggingfaceAudioLoader
from omegaconf import OmegaConf


conf = OmegaConf.load("test.yaml")

set_parallel(conf.parallel)


cutset = cutset_from_huggingface(data_info, len(ds['test']), "saeedq/librispeech_100h", "test")


os.environ["LHOTSE_AUDIO_BACKEND"] = "HuggingfaceDatasetsBackend"
# worker_plugin = HuggingfaceAudioLoader("saeedq/librispeech_100h", "test")


# with get_client(get_parallel_config()) as client:
#     client.register_worker_plugin(worker_plugin)
#     cuts_train = cutset.compute_and_store_features(Fbank(), "./tmp",num_jobs=16, executor=client)



# cuts_train = cutset.compute_and_store_features(Fbank(), "./tmp_1")

from lhotse.dataset.speech_recognition import K2SpeechRecognitionDataset
from lhotse.dataset.sampling import SimpleCutSampler
dataset = K2SpeechRecognitionDataset()

sampler = SimpleCutSampler(
    cuts=cutset,
    max_cuts=4,
    shuffle=True,
)



