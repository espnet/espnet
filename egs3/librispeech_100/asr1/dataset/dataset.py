import torch
from torch.utils.data import Dataset
from datasets import load_from_disk, Audio
import numpy as np


class LibriSpeechDataset(Dataset):
    def __init__(self, data_dir='data/', split=None):
        dataset_dict = load_from_disk(data_dir)

        if split:
            if split in dataset_dict:
                self.dataset = dataset_dict[split]
            else:
                raise ValueError(f"Split '{split}' not found in dataset.")
        else:
            self.dataset = dataset_dict

        # 'path' カラムを 'audio' 型にキャスト
        self.dataset = self.dataset.cast_column("audio", Audio())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        example = {
            "speech": item["audio"]["array"].astype(np.float32),
            "text": item["text"]
        }
        return example
    
    def get_text(self, idx):
        item = self.dataset[idx]
        return item["text"]
