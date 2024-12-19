import json
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from string import punctuation
from typing import List, Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset
from transformers import AutoTokenizer

SAMPLE_RATE = 16000
strip_punct_table = str.maketrans("", "", punctuation)


class ClothoMixupDataset(Dataset):
    def __init__(
        self,
        audio_dir: str,
        tokenizer_dir: str,
        chatgpt_augmented_data_path: str,
        chatgpt_rejected_data_path: str,
        split_name: Optional[str] = None,
        bypass_resample: Optional[bool] = False,
        lemma_file: Optional[str] = None,
        n_lemmas: Optional[int] = 2609,
        caption_embed_dir: Optional[str] = None,
        clap_embed_dir: Optional[str] = None,
        do_speed_augment: Optional[bool] = False,
        do_pitch_augment: Optional[bool] = False,
        speed_augment_range: Optional[List[float]] = [0.8, 1.2],
        pitch_augment_range: Optional[List[int]] = [-3, 4],
        max_audio_len: Optional[int] = 480000,
        do_mixup_normalize: Optional[bool] = False,
        sample_rate: Optional[int] = SAMPLE_RATE,
        audio_mixup_base_dir: Optional[str] = "local/clotho_audio_mixup",
    ):
        super().__init__()
        self.audio_mixup_base_dir = audio_mixup_base_dir
        self.sample_rate = sample_rate
        self.bypass_resample = bypass_resample
        self.audio_dir = audio_dir
        self.split_name = (
            split_name if split_name is not None else os.path.basename(audio_dir)
        )

        (self.idx_to_sample, self.idx_to_audio, self.captions) = self.get_captions(
            chatgpt_augmented_data_path,
            chatgpt_rejected_data_path,
        )

        self.tokenizer = self.get_tokenizer(tokenizer_dir)

        # lemma labels for encoder loss
        if lemma_file is not None:
            self.lemma_data = json.load(open(lemma_file))
            self.n_lemmas = n_lemmas
        else:
            self.lemma_data = None

        # text embedding for encoder loss
        self.caption_embed_dir = caption_embed_dir
        self.clap_embed_dir = clap_embed_dir

        # parameters for audio speed/pitch augmentation
        self.do_speed_augment = do_speed_augment
        self.do_pitch_augment = do_pitch_augment
        self.speed_augment_range = speed_augment_range
        self.pitch_augment_range = list(range(*pitch_augment_range))
        if self.do_pitch_augment:
            print(
                "[info] perform pitch shifting:", self.pitch_augment_range, "half notes"
            )
        self.max_audio_len = max_audio_len

        self.do_mixup_normalize = do_mixup_normalize

    def tokenize(self, text):
        text_id = self.tokenizer.encode(text, add_special_tokens=False)
        text_id += [self.tokenizer.eos_token_id]
        text_id = np.array(text_id, dtype=int)

        return text_id

    def get_captions(self, caption_json, rejected_json):
        rejected_df = json.load(open(rejected_json))
        caption_df = json.load(open(caption_json))["dataset"]

        all_rejects = set()
        seen_combs = set()
        for rej in rejected_df:
            all_rejects.add(tuple(rej["idx"]))

        idx_to_sample = []
        idx_to_audio = []
        captions = defaultdict(list)

        for i in range(len(caption_df)):
            # if i >= 100:
            #     break
            samp_a, samp_b = caption_df[i]["selected_pair"]
            audio_a, audio_b = caption_df[i]["audio_files"]

            assert (samp_a, samp_b) not in seen_combs
            seen_combs.add((samp_a, samp_b))

            for j in range(len(caption_df[i]["chatgpt_mixups"])):
                if (i, j) in all_rejects:
                    continue

                idx_to_sample.append((samp_a, samp_b))
                idx_to_audio.append((audio_a, audio_b))

                captions[(samp_a, samp_b)].append(
                    caption_df[i]["chatgpt_mixups"][j]
                    .strip()
                    .lower()
                    .translate(strip_punct_table)
                )

        return idx_to_sample, idx_to_audio, captions

    def get_tokenizer(self, tokenizer_dir):
        return AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

    def load_audio(self, audio_file):
        if not self.bypass_resample:
            wav, sr = librosa.load(
                audio_file,
                sr=self.sample_rate,
            )
            assert sr == self.sample_rate
        else:
            wav, sr = sf.read(audio_file)
            assert sr == 44100

        return wav, sr

    def mixup_audio(self, audio_a, audio_b, sample_name):
        mixed_audio_path = os.path.join(self.audio_mixup_base_dir, f"{sample_name}.wav")
        if not self.do_mixup_normalize:
            energy_a = np.mean(librosa.feature.rms(y=audio_a)[0] ** 2)
            energy_b = np.mean(librosa.feature.rms(y=audio_b)[0] ** 2)

            mix_db_ratio = np.random.uniform(-5, 5)
            mix_scale = np.sqrt(energy_a / (10 ** (mix_db_ratio / 10) * energy_b))
        else:
            import pyloudnorm as pyln

            mix_scale = 1
            base_db = np.random.uniform(-12, -5)
            mix_db_ratio = np.random.uniform(-5, 5)
            audio_a = pyln.normalize.peak(audio_a, base_db)
            audio_b = pyln.normalize.peak(audio_b, base_db + mix_db_ratio)

        if len(audio_a) > len(audio_b):
            st_idx = random.choice(range(len(audio_a) - len(audio_b) + 1))
            audio_a[st_idx : st_idx + len(audio_b)] += mix_scale * audio_b
            mixed_audio = audio_a
        else:
            st_idx = random.choice(range(len(audio_b) - len(audio_a) + 1))
            audio_b[st_idx : st_idx + len(audio_a)] += mix_scale * audio_a
            mixed_audio = audio_b

        try:
            sf.write(
                mixed_audio_path,
                mixed_audio,
                SAMPLE_RATE,
            )
        except Exception as e:
            print(f"Can not write audio {mixed_audio_path}")
            mixed_audio = None
            mixed_audio_path = None

        return mixed_audio, mixed_audio_path

    def augment_audio(self, audio):
        if self.do_pitch_augment and random.random() < 0.5:
            audio = librosa.effects.pitch_shift(
                audio,
                sr=self.sample_rate,
                n_steps=random.choice(self.pitch_augment_range),
            )

        if self.do_speed_augment and random.random() < 0.5:
            if audio.shape[0] / self.speed_augment_range[0] > self.max_audio_len:
                _min_speed = audio.shape[0] / self.max_audio_len
            else:
                _min_speed = self.speed_augment_range[0]

            assert audio.shape[0] / _min_speed <= self.max_audio_len + 1

            audio = librosa.effects.time_stretch(
                audio, rate=np.random.uniform(_min_speed, self.speed_augment_range[1])
            )

        return audio

    def read_lemma_labels(self, sample_name, sample_idx):
        lemmas = self.lemma_data[sample_name][sample_idx]["caption_kw_labels"]
        labels = np.zeros((self.n_lemmas,))

        for lem in lemmas:
            labels[lem] = 1

        return labels

    def read_caption_embed(self, sample_pair, sample_idx):
        sample_embed_dir = os.path.join(
            self.caption_embed_dir, str(sample_pair[0]) + "_" + str(sample_pair[1])
        )
        embed_choices = [
            f
            for f in os.listdir(sample_embed_dir)
            if f"chatgpt{sample_idx + 1:>02d}" in f
        ]
        # print(len(embed_choices))

        return np.load(os.path.join(sample_embed_dir, random.choice(embed_choices)))

    def read_clap_embed(self, sample_pair, sample_idx):
        sample_embed_dir = os.path.join(
            self.clap_embed_dir, str(sample_pair[0]) + "_" + str(sample_pair[1])
        )
        embed_choices = [
            f
            for f in os.listdir(sample_embed_dir)
            if f"chatgpt{sample_idx + 1:>02d}" in f
        ]
        # print(len(embed_choices))

        return np.load(os.path.join(sample_embed_dir, random.choice(embed_choices)))

    def __len__(self):
        return len(self.idx_to_sample)
        # return 10

    def __getitem__(self, index):
        samp_a, samp_b = self.idx_to_sample[index]
        audio_a, audio_b = self.idx_to_audio[index]
        audios_to_mix = []
        try:
            for audfile in [audio_a, audio_b]:
                audio, sr = self.load_audio(os.path.join(self.audio_dir, audfile))

                audios_to_mix.append(audio)
        except OSError as e:
            print("Skipping", samp_a, samp_b, "because", e)
            return None

        audio, audio_path = self.mixup_audio(
            audios_to_mix[0],
            audios_to_mix[1],
            sample_name=str(samp_a) + "_" + str(samp_b),
        )

        if (self.do_pitch_augment or self.do_speed_augment) and random.random() < 0.5:
            audio = self.augment_audio(audio)

        caption_idx = random.choice(range(len(self.captions[(samp_a, samp_b)])))

        caption = self.captions[(samp_a, samp_b)][caption_idx]
        # caption = self.tokenize(caption)

        bundle = {
            "sample_name": str(samp_a) + "_" + str(samp_b),
            "sr": sr,
            "audio_path": audio_path,
            "labels": caption,
            # "encoder_input": audio,
            # "attention_mask": np.full_like(audio, False, dtype=bool)
        }

        if self.lemma_data is not None:
            encoder_labels = self.read_lemma_labels((samp_a, samp_b), caption_idx)
            bundle["encoder_labels"] = encoder_labels
        elif self.caption_embed_dir is not None:
            encoder_embed = self.read_caption_embed((samp_a, samp_b), caption_idx)

            if self.clap_embed_dir is not None:
                clap_embed = self.read_clap_embed(
                    (samp_a, samp_b),
                    caption_idx,
                )
                encoder_embed = np.concatenate([encoder_embed, clap_embed], axis=-1)

            bundle["encoder_labels"] = encoder_embed
        else:
            bundle["encoder_labels"] = np.array([0])

        return bundle


mixup_captions_root_dir = sys.argv[1]
clotho_audio_base_dir = sys.argv[2]

# create directory paths for storing clotho mixup audio
# "downloads/CLOTHO_v2.1_audio_mixup/development/" < This is default.
audio_mixup_write_dir = f"{clotho_audio_base_dir}_audio_mixup/development/"
os.makedirs(os.path.dirname(audio_mixup_write_dir), exist_ok=True)

# Create mixup dataset
dset = ClothoMixupDataset(
    audio_dir=os.path.join(clotho_audio_base_dir, "clotho_audio_files/development"),
    tokenizer_dir="facebook/bart-base",
    chatgpt_augmented_data_path=os.path.join(
        mixup_captions_root_dir, "clotho_development_chatgpt_mixups.json"
    ),
    chatgpt_rejected_data_path=os.path.join(
        mixup_captions_root_dir, "clotho_development_chatgpt_mixups_err.json"
    ),
    # caption_embed_dir="sent_embedding/datasets/instructor-xl/clotho/development_chatgpt_mixup",
    # clap_embed_dir="sent_embedding/datasets/clap/clotho/development_chatgpt_mixup",
    audio_mixup_base_dir=audio_mixup_write_dir,
)

data_base = "data/development_clotho_chatgpt_mixup/"
output_txt_path = os.path.join(data_base, "text")
output_wav_scp_path = os.path.join(data_base, "wav.scp")
output_utt2spk_path = os.path.join(data_base, "utt2spk")

os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

start = time.time()


def _get_ith_item(i):
    data_i = dset[i]
    if data_i is None:
        return None, None, None
    uttid = f"clotho_chatmixup_{data_i['sample_name']}"
    return uttid, data_i["audio_path"], data_i["labels"]


results = None
with Pool() as pool:  # Use all cores
    results = pool.map(_get_ith_item, range(len(dset)))

assert results is not None

N_ERROR = 0
N_PROCESSED = 0
with open(output_txt_path, "w") as text_f, open(
    output_wav_scp_path, "w"
) as wav_scp_f, open(output_utt2spk_path, "w") as utt2spk_f:
    for i in range(len(dset)):
        uttid, audio_path, text = results[i]
        if uttid is None or len(uttid) == 0:
            N_ERROR += 1
            continue
        # Remove new lines and carriage returns
        text = text.replace("\n", " ").replace("\r", " ").strip()
        print(f"{uttid} {audio_path}", file=wav_scp_f)
        print(f"{uttid} {text}", file=text_f)
        print(f"{uttid} dummy", file=utt2spk_f)
        N_PROCESSED += 1
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1} mixup samples.")

print(f"Number of files skipped {N_ERROR}")
print(f"Number of files processed {N_PROCESSED}")
