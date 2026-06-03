"""
This script performs language identification (LID) on audio files and their
corresponding text transcriptions using public models.
Reference: Section 2.1.2 in the paper (https://arxiv.org/pdf/2506.00338)
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path

import fasttext
import librosa
import torch
from speechbrain.inference.classifiers import EncoderClassifier
from tqdm import tqdm

from espnet2.legacy.nets.pytorch_backend.nets_utils import pad_list

logging.basicConfig()
logging.getLogger().setLevel(level=logging.ERROR)


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        self.audios = {}
        self.data = []
        with open(data_file, "r") as fin:
            for line in fin:
                sample = json.loads(line.strip())
                self.data.append(
                    {
                        "utt_id": sample["utt_id"],
                        "wav_path": sample["wav_path"],
                        "start_time": sample["start_time"],
                        "end_time": sample["end_time"],
                        "lang": sample["lang"],
                        "text": sample["text"],
                        "prev_text": sample["prev_text"],
                        "confidences": sample["confidences"],
                    }
                )
        print(f"------------ Loaded {len(self.data)} samples ------------")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        fs = 16000  # the model is trained on 16k

        if sample["wav_path"] not in self.audios:
            self.audios[sample["wav_path"]] = librosa.load(sample["wav_path"], sr=fs)[0]

        wav = self.audios[sample["wav_path"]][
            int(sample["start_time"] * fs) : int(sample["end_time"] * fs)
        ]
        if len(wav) == 0:
            return None

        res = {
            "utt_id": sample["utt_id"],
            "wav": torch.tensor(wav, dtype=torch.float32),
            "lang": sample["lang"],
            "text": sample["text"],
            "prev_text": sample["prev_text"],
            "confidences": sample["confidences"],
        }
        return res


def collate_fn(batch):
    batch = [x for x in batch if x is not None]

    utt_ids = [x["utt_id"] for x in batch]
    langs = [x["lang"] for x in batch]
    texts = [x["text"] for x in batch]
    prev_texts = [x["prev_text"] for x in batch]

    wavs = [x["wav"] for x in batch]
    wav_lens = torch.tensor([len(x) for x in wavs])
    wavs = pad_list(wavs, 0.0)  # (B, T)
    wav_lens = wav_lens / wav_lens.max()  # (B,)

    scores = [x["confidences"] for x in batch]

    return utt_ids, wavs, wav_lens, langs, texts, prev_texts, scores


def lid_fasttext(model, text, max_len=None):
    if max_len is None:
        max_len = len(text)
    lang = model.predict(text[:max_len])[0][0].removeprefix("__label__")
    return lang


def lid_speechbrain_batched(model, wavs, wav_lens):
    predictions = model.classify_batch(wavs, wav_lens)[3]  # list of label texts
    return [p.split(":")[0] for p in predictions]


def process(in_file, out_file, fasttext_ori):
    with tempfile.TemporaryDirectory() as tmpdir, open(out_file, "w") as fout:
        sb_model = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir=tmpdir,
            run_opts={"device": "cuda"},
        )
        dataset = SpeechDataset(in_file)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=72,  # smaller than (2^31 - 1) / (9216 * 3001) = 77
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=8,
        )

        try:
            for utt_ids, wavs, wav_lens, langs, texts, prev_texts, scores in tqdm(
                dataloader, miniters=1, maxinterval=600, mininterval=10
            ):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    speech_preds = lid_speechbrain_batched(sb_model, wavs, wav_lens)
                    text_preds = [lid_fasttext(fasttext_ori, text) for text in texts]
                    prev_text_preds = []
                    for prev_text in prev_texts:
                        if prev_text == "<na>":
                            prev_text_preds.append("<na>")
                        else:
                            prev_text_preds.append(
                                lid_fasttext(fasttext_ori, prev_text)
                            )

                for (
                    utt_id,
                    lang,
                    text,
                    prev_text,
                    speech_pred,
                    text_pred,
                    prev_text_pred,
                    score,
                ) in zip(
                    utt_ids,
                    langs,
                    texts,
                    prev_texts,
                    speech_preds,
                    text_preds,
                    prev_text_preds,
                    scores,
                ):
                    res = {
                        "utt_id": utt_id,
                        "lang": lang[1:-1],
                        "speech_pred": speech_pred,
                        "text_pred": text_pred,
                        "prev_text_pred": prev_text_pred,
                        "text": text,
                        "prev_text": prev_text,
                        "confidences": score,
                    }
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"Error: {e}")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    fasttext_ori = fasttext.load_model("models/fasttext/lid.176.bin")

    process(args.in_file, args.out_file, fasttext_ori)
