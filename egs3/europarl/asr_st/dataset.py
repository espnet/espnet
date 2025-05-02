import torch
from torch.utils.data import Dataset
from datasets import load_from_disk, Audio
import numpy as np
import librosa
from espnet2.bin.s2t_inference import Speech2Text


class EuroparlASRDataset(Dataset):
    def __init__(self, data_dir="data", split=None):
        dataset_dict = load_from_disk(data_dir)

        if split:
            if split in dataset_dict:
                self.dataset = dataset_dict[split]
            else:
                raise ValueError(f"Split '{split}' not found in dataset.")
        else:
            self.dataset = dataset_dict

        self.iso_code = {
            "de": "deu",
            "fr": "fra",
            "it": "ita",
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_lang = item['src_lang']
        src_text = item[f"text.{src_lang}"]
        example = {
            "speech": librosa.load(item["audio"]["path"], sr=16000, mono=True,)[0].astype(np.float32),
            "text": f'<{self.iso_code[src_lang]}><asr><notimestamps> {src_text}',
            "text_ctc": src_text,
            "text_prev": "<na>",
        }
        return str(idx), example



class EuroparlSTDataset(Dataset):
    def __init__(self, data_dir="data", split=None):
        dataset_dict = load_from_disk(data_dir)

        if split:
            if split in dataset_dict:
                self.dataset = dataset_dict[split]
            else:
                raise ValueError(f"Split '{split}' not found in dataset.")
        else:
            self.dataset = dataset_dict

        self.iso_code = {
            "de": "deu",
            "fr": "fra",
            "it": "ita",
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_lang = item['src_lang']
        tgt_lang = item['src_lang']
        src_text = item[f"text.{src_lang}"]
        tgt_text = item[f"text.{tgt_lang}"]
        example = {
            "speech": librosa.load(
                item["audio"]["path"], sr=None, mono=True,
            )[0].astype(np.float32),
            "text": f'<{self.iso_code[src_lang]}><st_{self.iso_code[tgt_lang]}><notimestamps> {tgt_text}',
            "text_ctc": src_text,
            "text_prev": "<na>",
        }
        return str(idx), example



class OWSMTokenizeTransform:
    def __init__(self, model_tag):
        owsm_model = Speech2Text.from_pretrained(model_tag)
        self.tokenizer = owsm_model.tokenizer
        self.converter = owsm_model.converter
    
    def tokenize(self, text):
        return np.array(self.converter.tokens2ids(self.tokenizer.text2tokens(text)))
    
    def __call__(self, data):
        idx, example = data
        ret = dict(
            speech=example['speech'],
            text=self.tokenize(example['text']),
            text_ctc=self.tokenize(example['text_ctc']),
            text_prev=self.tokenize(example['text_prev']),
        )
        return (idx, ret)
