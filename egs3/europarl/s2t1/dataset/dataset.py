import contextlib
import os
import sys

import numpy as np
from lhotse import CutSet
from torch.utils.data import Dataset

from espnet2.bin.s2t_inference import Speech2Text


class EuroparlASRDataset(Dataset):
    def __init__(self, data_dir="data", split=None):
        if split is None:
            raise ValueError("ASRDataset requires a split name (e.g., 'dev', 'train')")
        path = f"{data_dir}/{split}.jsonl.gz"
        self.cuts = CutSet.from_file(path)

        self.iso_code = {
            "de": "deu",
            "fr": "fra",
            "it": "ita",
        }

    def __len__(self):
        return len(self.cuts)

    def __getitem__(self, idx):
        cut = self.cuts[idx]
        audio = cut.load_audio()[0]

        supervision = cut.supervisions[0]

        src_lang = supervision.custom['src_lang']
        src_text = supervision.text.lower()

        example = {
            "speech": audio.astype(np.float32),
            "text": f'<{self.iso_code[src_lang]}><asr><notimestamps> {src_text}',
            "text_ctc": src_text,
            "text_prev": "<na>",
        }
        return example
    
    def get_text(self, idx):
        cut = self.cuts[idx]
        supervision = cut.supervisions[0]
        return supervision.text.lower()


class EuroparlSTDataset(Dataset):
    def __init__(self, data_dir="data", split=None):
        if split is None:
            raise ValueError("STDataset requires a split name (e.g., 'dev', 'train')")
        path = f"{data_dir}/{split}.jsonl.gz"
        self.cuts = CutSet.from_file(path)

        self.iso_code = {
            "de": "deu",
            "fr": "fra",
            "it": "ita",
        }

    def __len__(self):
        return len(self.cuts)

    def __getitem__(self, idx):
        cut = self.cuts[idx]
        audio = cut.load_audio()[0]
        supervision = cut.supervisions[0]
        custom = supervision.custom

        src_lang = custom['src_lang']
        tgt_lang = custom['tgt_lang']
        src_text = supervision.text.lower()
        tgt_text = custom[f"text.{tgt_lang}"].lower()

        example = {
            "speech": audio.astype(np.float32),
            "text": f'<{self.iso_code[src_lang]}><st_{self.iso_code[tgt_lang]}>'
            + f'<notimestamps> {tgt_text}',
            "text_ctc": src_text,
            "text_prev": "<na>",
        }
        return example

    def get_text(self, idx):
        cut = self.cuts[idx]
        supervision = cut.supervisions[0]
        custom = supervision.custom
        tgt_lang = custom['tgt_lang']
        return custom[f"text.{tgt_lang}"].lower()


class OWSMTokenizeTransform:
    def __init__(self, model_tag):
        owsm_model = Speech2Text.from_pretrained(model_tag)
        self.tokenizer = owsm_model.tokenizer
        self.converter = owsm_model.converter

    def tokenize(self, text):
        return np.array(self.converter.tokens2ids(self.tokenizer.text2tokens(text)))

    def __call__(self, data):
        example = data
        ret = dict(
            speech=example['speech'],
            text=self.tokenize(example['text']),
            text_ctc=self.tokenize(example['text_ctc']),
            text_prev=self.tokenize(example['text_prev']),
        )
        return ret
