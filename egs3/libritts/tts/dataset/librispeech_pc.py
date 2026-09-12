"""LibriSpeech-PC cross-sentence eval dataset with explicit prompt pairs.

Unlike ``LibriTTSDataset``'s ``ref_mode`` (random same/cross-speaker prompt
selection), every row of the manifest built by
``local/prepare_librispeech_pc.py`` pins its prompt, reproducing the paper's
fixed pairing. Output keys mirror the ``inference + ref_mode`` path of
``dataset/dataset.py`` so ``inference_f5*.yaml`` input_key wiring and
``src.inference.build_output`` work unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset as TorchDataset


class LibriSpeechPCDataset(TorchDataset):
    def __init__(
        self,
        manifest_path: str | Path,
        fs: int | None = 24000,
    ) -> None:
        self.fs = fs
        self.rows: list[tuple[str, str, str, str, str]] = []
        with Path(manifest_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                gen_utt, gen_text, ref_utt, ref_wav, ref_text = line.split("\t")
                self.rows.append((gen_utt, gen_text, ref_utt, ref_wav, ref_text))
        if not self.rows:
            raise RuntimeError(f"Empty manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        gen_utt, gen_text, _ref_utt, ref_wav, ref_text = self.rows[idx]
        speech, sr = sf.read(ref_wav, dtype="float32")
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if self.fs is not None and sr != self.fs:
            speech = torchaudio.functional.resample(
                torch.from_numpy(speech), sr, self.fs
            ).numpy()
        return {
            "utt_id": gen_utt,
            "text": gen_text,
            "raw_text": gen_text,
            "ref_speech": np.asarray(speech, dtype=np.float32),
            "ref_text": ref_text,
            "ref_wav_path": str(ref_wav),
        }


# Alias consumed by
# espnet3.components.data.dataset_module.instantiate_dataset_reference, which
# does `getattr(module, "Dataset")` after `import_module(data_src)` when a
# test split's `data_src` is set to this module's dotted path
Dataset = LibriSpeechPCDataset
