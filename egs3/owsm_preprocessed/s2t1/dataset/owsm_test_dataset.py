import torch
from torch.utils.data import Dataset
from datasets import load_from_disk, Audio
import numpy as np


class OWSMTestSpeechDataset(Dataset):
    def __init__(self,
                 data_dir='data/', split=None, lang_sym="<eng>", task_sym="<asr>"):
        print(data_dir, split)
        dataset_dict = load_from_disk(data_dir)
        self.lang_sym = lang_sym
        self.task_sym = task_sym

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
            "text": item["text"],
            "lang_sym": self.lang_sym,
            "task_sym": self.task_sym,
        }
        return example
    
    def get_text(self, idx):
        item = self.dataset[idx]
        return item["text"]
