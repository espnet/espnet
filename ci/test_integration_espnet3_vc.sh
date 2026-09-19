#!/usr/bin/env bash

# Integration test for the ESPnet3 VC system (VCSystem + kNN-VC components).
#
# The shipped VC recipe (egs3/librispeech_100/vc) needs LibriSpeech and the
# 1.2 GB WavLM-Large checkpoint, so this leg builds a miniature recipe instead:
# the mini_an4 audio fixture (3 speakers, 16 kHz, already in the repo), a stub
# encoder standing in for WavLM, and a deliberately tiny HiFi-GAN. It exercises
# the same code paths as a real run -- run.py stage dispatch, VCSystem,
# PrepareFeaturesProvider/Runner, KNNVCVocoderModel training through Lightning's
# multi-optimizer path, and KNNVCModel inference -- end to end:
#
#   create_dataset -> prepare_features -> train -> infer

set -euo pipefail

. tools/activate_python.sh
. tools/extra_path.sh

python="coverage run --append"
cwd=$(pwd)

gen_dummy_coverage(){
    touch empty.py
    ${python} empty.py
}

python3 -m pip install -e .

echo "==== [ESPnet3] VC (kNN-VC) ===="
gen_dummy_coverage

# mini_an4 ships its corpus as a tarball so CI never hits the network.
an4_root="${cwd}/egs3/mini_an4/asr/downloads"
if [ ! -d "${an4_root}/an4/wav/an4_clstk" ]; then
    tar -xzf "${cwd}/egs3/mini_an4/asr/downloads.tar.gz" -C "${cwd}/egs3/mini_an4/asr"
fi

work_dir=$(mktemp -d)
trap 'rm -rf "${work_dir}"' EXIT
mkdir -p "${work_dir}/conf"

# --- stub encoder: deterministic, tiny, and no download -------------------
cat > "${work_dir}/stub_encoder.py" <<'PYEOF'
"""Stub SSL encoder for the VC integration test (stands in for WavLM)."""

import numpy as np
import torch


class StubEncoder(torch.nn.Module):
    """Project fixed-size waveform windows to a small feature vector."""

    sample_rate = 16000
    hop_length = 80

    def __init__(self, output_dim=32, device="cpu", **_kwargs):
        super().__init__()
        self._output_dim = int(output_dim)
        generator = torch.Generator().manual_seed(0)
        self.projection = torch.nn.Parameter(
            torch.randn(self.hop_length, self._output_dim, generator=generator),
            requires_grad=False,
        )
        self.to(torch.device(device))

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def device(self):
        return self.projection.device

    @torch.inference_mode()
    def encode(self, speech, pad_to_hop=False):
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        speech = speech.to(self.device, dtype=torch.float32).flatten()
        if pad_to_hop:
            remainder = speech.numel() % self.hop_length
            speech = torch.nn.functional.pad(
                speech, (0, self.hop_length - remainder)
            )
        n_frames = speech.numel() // self.hop_length
        frames = speech[: n_frames * self.hop_length].view(n_frames, self.hop_length)
        return frames @ self.projection

    def forward(self, speech):
        return self.encode(speech)
PYEOF

# --- dataset: the three an4 speaker directories, in the three VC kinds ----
cat > "${work_dir}/vc_dataset.py" <<'PYEOF'
"""Miniature VC dataset over the mini_an4 audio fixture."""

import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from espnet3.components.data.dataset_builder import DatasetBuilder

HOP_LENGTH = 80
AUDIO_ROOT = Path(os.environ["ESPNET3_VC_TEST_AUDIO"])


def _utterances():
    return sorted(AUDIO_ROOT.glob("*/*.sph"))


def _read(path):
    array, sample_rate = sf.read(str(path), dtype="float32")
    assert sample_rate == 16000, path
    return np.ascontiguousarray(array)


class MiniVCBuilder(DatasetBuilder):
    """Validate that the extracted an4 fixture is present."""

    def is_source_prepared(self, **_kwargs):
        return len(_utterances()) > 0

    def prepare_source(self, **_kwargs):
        if not self.is_source_prepared():
            raise FileNotFoundError(f"No .sph files under {AUDIO_ROOT}")

    def is_built(self, **_kwargs):
        return self.is_source_prepared()

    def build(self, **_kwargs):
        self.prepare_source()


class MiniVCDataset(TorchDataset):
    """Audio / vocoder / conversion views over the fixture, as in the recipe."""

    def __init__(
        self,
        kind="audio",
        features_dir=None,
        segment_frames=8,
        num_pairs=2,
        seed=0,
        **_kwargs,
    ):
        self.kind = kind
        self.features_dir = Path(features_dir) if features_dir else None
        self.segment_frames = segment_frames
        self.paths = _utterances()
        if kind == "conversion":
            rng = random.Random(seed)
            speakers = sorted({p.parent.name for p in self.paths})
            pairs = []
            for path in self.paths:
                targets = [s for s in speakers if s != path.parent.name]
                pairs.append((path, rng.choice(targets)))
            self.pairs = sorted(pairs, key=lambda pair: (pair[1], pair[0].name))[
                :num_pairs
            ]

    # prepare_features contract
    def get_pool_key(self, idx):
        return self.paths[int(idx)].parent.name

    def get_feature_name(self, idx):
        path = self.paths[int(idx)]
        return f"{path.parent.name}/{path.stem}"

    def __len__(self):
        return len(self.pairs) if self.kind == "conversion" else len(self.paths)

    def __getitem__(self, idx):
        if self.kind == "conversion":
            path, target = self.pairs[int(idx)]
            references = [p for p in self.paths if p.parent.name == target]
            return {
                "speech": _read(path),
                "reference_speech": [_read(p) for p in references],
                "target_speaker": target,
                "pair_id": f"{path.stem}_to_{target}",
            }

        path = self.paths[int(idx)]
        speech = _read(path)
        if self.kind == "audio":
            return {"speech": speech}

        feats = np.load(
            self.features_dir / f"{path.parent.name}/{path.stem}.npy"
        ).astype(np.float32)
        n_frames = min(feats.shape[0], speech.shape[0] // HOP_LENGTH)
        feats, speech = feats[:n_frames], speech[: n_frames * HOP_LENGTH]
        seg = self.segment_frames
        if n_frames > seg:
            start = random.randint(0, n_frames - seg)
            feats = feats[start : start + seg]
            speech = speech[start * HOP_LENGTH : (start + seg) * HOP_LENGTH]
        elif n_frames < seg:
            feats = np.pad(feats, ((0, seg - n_frames), (0, 0)))
            speech = np.pad(speech, (0, seg * HOP_LENGTH - speech.shape[0]))
        return {"feats": feats, "speech": speech}


Dataset = MiniVCDataset
DatasetBuilder = MiniVCBuilder
PYEOF

cat > "${work_dir}/run.py" <<'PYEOF'
from egs3.TEMPLATE.vc.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.systems.vc.system import VCSystem

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=VCSystem, stages=stages_to_run)
PYEOF

encoder_target="stub_encoder.StubEncoder"
cat > "${work_dir}/conf/training.yaml" <<PYEOF
recipe_dir: ${work_dir}
exp_tag: vc_integration
num_device: 1

create_dataset:
  recipe_dir: \${recipe_dir}

prepare_features:
  features_dir: \${recipe_dir}/data/features
  dataset:
    - name: train
      data_src: vc_dataset
      data_src_args:
        kind: audio
  encoder:
    _target_: ${encoder_target}
    output_dim: 32
  prematch: true
  topk: 2
  device: cpu

dataset:
  train:
    - data_src: vc_dataset
      data_src_args:
        kind: vocoder
        features_dir: \${prepare_features.features_dir}
        segment_frames: 8
  valid:
    - data_src: vc_dataset
      data_src_args:
        kind: vocoder
        features_dir: \${prepare_features.features_dir}
        segment_frames: 8

model:
  _target_: espnet3.systems.vc.models.knnvc.vocoder.KNNVCVocoderModel
  generator:
    in_channels: 32
    projection_channels: 16
    channels: 16
    upsample_scales: [5, 4, 2, 2]
    upsample_kernel_sizes: [10, 8, 4, 4]
    resblock_kernel_sizes: [3]
    resblock_dilations: [[1, 2]]
  discriminator:
    scales: 1
    periods: [2]
    scale_discriminator_params:
      channels: 4
      max_downsample_channels: 8
      max_groups: 2
      downsample_scales: [2, 1]
    period_discriminator_params:
      channels: 4
      max_downsample_channels: 8
      downsample_scales: [2, 1]
  mel:
    n_fft: 256
    hop_length: 80
    win_length: 256
    n_mels: 16

optimizers:
  generator:
    optimizer: {_target_: torch.optim.AdamW, lr: 0.0002}
    params: generator
  discriminator:
    optimizer: {_target_: torch.optim.AdamW, lr: 0.0002}
    params: discriminator

schedulers:
  generator:
    scheduler: {_target_: torch.optim.lr_scheduler.ExponentialLR, gamma: 0.999}
    interval: epoch
  discriminator:
    scheduler: {_target_: torch.optim.lr_scheduler.ExponentialLR, gamma: 0.999}
    interval: epoch

dataloader:
  train: {batch_size: 2, num_workers: 0}
  valid: {batch_size: 1, num_workers: 0}

trainer:
  accelerator: cpu
  devices: 1
  max_epochs: 1
  limit_train_batches: 2
  limit_val_batches: 1
  log_every_n_steps: 1
PYEOF

cat > "${work_dir}/conf/inference.yaml" <<PYEOF
recipe_dir: ${work_dir}

dataset:
  test:
    - name: test
      data_src: vc_dataset
      data_src_args:
        kind: conversion
        num_pairs: 2

model:
  _target_: espnet3.systems.vc.models.knnvc.model.KNNVCModel
  vocoder_checkpoint: \${exp_dir}/last.ckpt
  encoder:
    _target_: ${encoder_target}
    output_dim: 32
  generator:
    in_channels: 32
    projection_channels: 16
    channels: 16
    upsample_scales: [5, 4, 2, 2]
    upsample_kernel_sizes: [10, 8, 4, 4]
    resblock_kernel_sizes: [3]
    resblock_dilations: [[1, 2]]
  topk: 2
  device: cpu

input_key: [speech, reference_speech, target_speaker]
PYEOF

cd "${work_dir}" || exit
ESPNET3_VC_TEST_AUDIO="${an4_root}/an4/wav/an4_clstk" \
PYTHONPATH="${work_dir}:${cwd}:${PYTHONPATH:-}" \
    ${python} run.py \
        --stages create_dataset prepare_features train infer \
        --training_config conf/training.yaml \
        --inference_config conf/inference.yaml

# The stage outputs must actually exist: features per utterance, a checkpoint,
# and one converted WAV per conversion pair.
test -s "${work_dir}/data/features/feats.train.scp"
test "$(find "${work_dir}/data/features" -name '*.npy' | wc -l)" -ge 3
test -e "${work_dir}/exp/vc_integration/last.ckpt"
test -s "${work_dir}/exp/vc_integration/inference/test/wav.scp"
test "$(find "${work_dir}/exp/vc_integration/inference/test" -name '*.wav' | wc -l)" -eq 2
echo "==== [ESPnet3] VC integration test passed ===="

cd "${cwd}" || exit
