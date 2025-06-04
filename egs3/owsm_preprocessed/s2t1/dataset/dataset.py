from datasets import load_from_disk
from espnet3.data.dataset import ShardedDataset
from pathlib import Path
import kaldiio
import numpy as np


class OWSMV3Dataset(ShardedDataset):
    def __init__(self, dataset_path="data", split="train", initialize_on_shard=False):
        """
        Args:
            dataset_path (str or Path): path to the saved DatasetDict or shard
            split (str): one of 'dev', 'train_v3', etc.
            shard_id (int, optional): for sharded splits (e.g. train_v3/0)
        """
        self.split = split
        self.dataset_path = dataset_path
        if initialize_on_shard:
            self.dataset = []
        else:
            self.dataset = load_from_disk(Path(self.dataset_path) / split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        return {
            "speech": kaldiio.load_mat(example["audio_path"])[1].astype(np.float32),
            "text": example["text"],
            "text_ctc": example["text_ctc"],
            "text_prev": example["text_prev"],
        }

    def shard(self, shard_id: int):
        return OWSMV3Dataset(
            dataset_path=str(self.dataset_path),
            split=self.split + f"/{shard_id}",
            initialize_on_shard=False,
        )

    def get_text(self, idx):
        return self.dataset[idx]['text_ctc']
