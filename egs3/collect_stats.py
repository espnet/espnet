
# Data preparation
from collections import defaultdict
from lhotse.supervision import SupervisionSegment
from lhotse.audio import Recording
from espnetez.data.lhotse_utils import HuggingFaceAudioSource, HuggingfaceDatasetsBackend, cutset_from_huggingface
from datasets import load_dataset, load_from_disk
from lhotse.audio.backend import set_current_audio_backend
from lhotse.cut import MonoCut
from lhotse import CutSet, Mfcc, Fbank
import os
from dask.distributed import get_worker
from dask.distributed import WorkerPlugin, as_completed
import datasets
import torch
import numpy as np

ds = load_from_disk("/home/msomeki/workspace/librispeech_dataset")
set_current_audio_backend(HuggingfaceDatasetsBackend(
    "/home/msomeki/workspace/librispeech_dataset"
))


## parallel
from espnetez.parallel import parallel_map, set_parallel, get_client, get_parallel_config
from espnetez.data.lhotse_utils import HuggingfaceAudioLoader
from omegaconf import OmegaConf
from hydra.utils import instantiate

from tqdm import tqdm


conf = OmegaConf.load("test.yaml")
set_parallel(conf.parallel)


class CollectStatsPlugin(WorkerPlugin):
    def __init__(self, frontend_config, dataset_id, split: str = None) -> None:
        super().__init__()
        self.dataset_id = dataset_id
        self.split = split
        self.frontend_config = frontend_config
    def setup(self, worker):
        worker.dataset = self.load_dataset()
        # worker.dataset = _load_from_hf_or_disk(self.dataset_id)
        worker.dataset_id = self.dataset_id
        worker.split = self.split
        worker.frontend = instantiate(self.frontend_config)
    def load_dataset(self):
        try:
            return datasets.load_dataset(
                self.dataset_id
            )
        except:
            return datasets.load_from_disk(
                self.dataset_id
            )



def runner(idx):
    ds = get_worker().dataset
    data = ds['train-clean-100'][idx]
    audio_length = torch.LongTensor([data["audio"]["array"].shape[0]])
    feat, feat_len = get_worker().frontend(
        torch.from_numpy(data['audio']['array'])[None].type(torch.float32),
        audio_length,
    )
    return feat.sum(), (feat ** 2).sum(), feat_len


feat_stats = 0
feat_sq_stats = 0
count_stats = 0
worker_plugin = CollectStatsPlugin(
    conf.frontend,
    "/home/msomeki/workspace/librispeech_dataset",
)
with get_client(get_parallel_config()) as client:
    client.register_plugin(worker_plugin)
    futures = client.map(runner, range(len(ds['train-clean-100'])))
    for future in tqdm(as_completed(futures), total=len(futures)):
        feat, feat_sq , count = future.result()
        feat_stats += feat
        feat_sq_stats += feat_sq
        count_stats += count


np.savez(
    "train_feat_stats.npz",
    count=count_stats,
    sum=feat_stats,
    sum_square=feat_sq_stats,
)


