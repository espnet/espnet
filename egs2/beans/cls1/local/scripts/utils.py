import hashlib
import math
import os
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import Dataset


def get_wav_length_in_secs(path):
    info = torchaudio.info(path)
    return info.num_frames / info.sample_rate


def get_md5(file_name):
    with open(file_name, mode="rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)

        return file_hash.hexdigest()


def check_md5(file_name, md5):
    if md5 != get_md5(file_name):
        assert False, f"md5 for {file_name} does not match"


def divide_waveform_to_chunks(path, target_dir, chunk_size, target_sample_rate=16000):
    waveform, sample_rate = sf.read(path)
    waveform = torch.tensor(waveform, dtype=torch.float32)
    assert (
        waveform.min() >= -1.0 and waveform.max() <= 1.0
    ), f"waveform should be normalized. min: {waveform.min()}, max: {waveform.max()}"
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, n_samples)
    waveform = waveform.mean(dim=0).unsqueeze(0)  # single channel always

    if sample_rate != target_sample_rate:
        transform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = transform(waveform)

    num_samples = waveform.shape[1]
    num_seconds = num_samples / target_sample_rate
    num_chunks = math.ceil(num_seconds / chunk_size)
    target_paths = []
    for chunk in range(num_chunks):
        target_path = Path(target_dir) / f"{Path(path).stem}.{chunk:03d}.wav"
        st_sample = int(chunk * chunk_size * target_sample_rate)
        ed_sample = int((chunk + 1) * chunk_size * target_sample_rate)
        sf.write(
            target_path,
            waveform[:, st_sample:ed_sample].squeeze(0).numpy(),
            target_sample_rate,
        )
        target_paths.append(str(target_path))

    return target_paths


def divide_annotation_to_chunks(annotations, chunk_size):
    chunks = defaultdict(list)
    for anon in annotations:
        st, ed = anon["st"], anon["ed"]  # in seconds
        st_chunk, ed_chunk = int(st // chunk_size), int(ed // chunk_size)

        for chunk in range(st_chunk, ed_chunk + 1):
            if chunk == st_chunk and chunk == ed_chunk:
                local_st, local_ed = st - chunk * chunk_size, ed - chunk * chunk_size
            elif chunk == st_chunk:
                local_st, local_ed = st - chunk * chunk_size, chunk_size
            elif chunk == ed_chunk:
                local_st, local_ed = 0, ed - chunk * chunk_size
            else:
                local_st, local_ed = 0, chunk_size

            new_anon = dict(anon)
            new_anon["st"], new_anon["ed"] = local_st, local_ed
            chunks[chunk].append(new_anon)

    return chunks


def _get_waveform(filename, max_duration, target_sample_rate=16000):
    waveform, sample_rate = sf.read(filename)
    # (1, n_samples)
    waveform = torch.tensor(waveform, dtype=torch.float64).unsqueeze(0)

    if sample_rate != target_sample_rate:
        # This should not be needed since we resampled earlier
        transform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = transform(waveform)

    max_samples = max_duration * target_sample_rate
    waveform = waveform[0, :max_samples]
    if waveform.shape[0] < max_samples:
        waveform = F.pad(waveform, (0, max_samples - waveform.shape[0]))

    return waveform


class BeansRecognitionDataset(Dataset):
    """Constructs sound event detection training and evaluation
    samples from the provided dataset, in the style of BEANS benchmark.
    """

    def __init__(
        self,
        dataset,
        window_width,
        window_shift,
        target_dir,
        sample_rate=16_000,
    ):
        self.sample_rate = sample_rate
        self.max_duration = 60  # sec
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)

        self.xs = []  # wav paths
        self.ys = []  # labels (text)

        for data in dataset:
            wav_path = data["path"]
            length = data["length"]  # sec

            num_windows = int((length - window_width) / window_shift) + 1

            for window_id in range(num_windows):
                st, ed = (
                    window_id * window_shift,
                    window_id * window_shift + window_width,
                )
                offset_st, offset_ed = st * self.sample_rate, ed * self.sample_rate
                self.xs.append((wav_path, offset_st, offset_ed))

                y = set()
                for anon in data["annotations"]:
                    label_name = anon["label"].strip().lower()

                    if (st <= anon["st"] <= ed) or (st <= anon["ed"] <= ed):
                        denom = min(ed - st, anon["ed"] - anon["st"])
                        if denom == 0:
                            continue
                        overlap = (min(ed, anon["ed"]) - max(st, anon["st"])) / denom
                        if overlap > 0.2:
                            y.add(label_name)
                    if anon["st"] <= st and ed <= anon["ed"]:
                        y.add(label_name)

                self.ys.append(list(y))

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        """Returns path to wav file and label text."""
        wav_path, offset_st, offset_ed = self.xs[idx]
        x = _get_waveform(
            wav_path,
            max_duration=self.max_duration,
            target_sample_rate=self.sample_rate,
        )
        x = x[offset_st:offset_ed]
        # Write the smaller chunk to disk and return its path.
        # Yes this is wasteful but ok.
        smaller_chunk_path = (
            Path(self.target_dir) / f"{Path(wav_path).stem}.window{idx:09d}.wav"
        )
        sf.write(smaller_chunk_path, x.numpy(), self.sample_rate)
        return {"path": smaller_chunk_path, "label": self.ys[idx]}
