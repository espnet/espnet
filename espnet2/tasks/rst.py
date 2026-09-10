"""Task definition for the ESPnet-Sidon feature predictor."""

import logging
import math
import os
import random
import zlib
from typing import List, Tuple

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

from espnet2.enh.sidon_model import (
    SSL_ENCODERS,
    SidonFeaturePredictor,
    build_ssl_encoder,
)
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.trainer import Trainer
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool

logger = logging.getLogger(__name__)


def _audio_files(directory: str) -> List[str]:
    if not directory or not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith((".wav", ".flac"))
    )


def _load_mono(path: str) -> Tuple[torch.Tensor, int]:
    """Read a WAV as a mono float tensor with soundfile.

    torchaudio's loader dispatches to TorchCodec from torchaudio 2.9 on, which
    is not an ESPnet dependency; soundfile is.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(audio.mean(axis=1)), sr


def _reverb(wav: torch.Tensor, sr: int, files: List[str]) -> torch.Tensor:
    if not files:
        return wav
    rir, rir_sr = _load_mono(random.choice(files))
    if rir_sr != sr:
        rir = AF.resample(rir, rir_sr, sr)
    rir = rir / rir.abs().max().clamp_min(1e-8)
    # Discard the propagation delay before the direct path. Convolving with a
    # RIR that starts with silence shifts the whole signal later, but
    # speech_ref1 is not shifted, and SidonFeaturePredictor compares features
    # at matching frame indices -- so the delay would appear as a permanent
    # misalignment between the degraded input and its own target.
    peak = int(torch.argmax(rir.abs()).item())
    rir = rir[peak:]
    rir = rir[: max(1, wav.numel() // 2)]
    if rir.numel() == 0:
        return wav
    return AF.fftconvolve(wav[None], rir[None])[0, : wav.numel()]


def _noise(wav: torch.Tensor, sr: int, files: List[str]) -> torch.Tensor:
    if not files:
        return wav
    noise, noise_sr = _load_mono(random.choice(files))
    if noise_sr != sr:
        noise = AF.resample(noise, noise_sr, sr)
    if not noise.numel():
        return wav
    noise = noise.repeat(math.ceil(wav.numel() / noise.numel()))[: wav.numel()]
    snr = torch.tensor([random.uniform(-5, 20)])
    return AF.add_noise(wav[None], noise[None], snr)[0]


def _band_limit(wav: torch.Tensor, sr: int) -> torch.Tensor:
    target = random.choice([8000, 16000, 22050, 24000, 44100, 48000])
    if target == sr:
        return wav
    return AF.resample(AF.resample(wav, sr, target), target, sr)[: wav.numel()]


def _clip(wav: torch.Tensor) -> torch.Tensor:
    low = torch.quantile(wav, random.uniform(0.0, 0.1))
    high = torch.quantile(wav, random.uniform(0.9, 1.0))
    return wav.clamp(low, high) if low < high else wav


def _codec(wav: torch.Tensor, sr: int) -> torch.Tensor:
    effect = torchaudio.io.AudioEffector(
        format="mp3",
        codec_config=torchaudio.io.CodecConfig(qscale=random.randint(1, 10)),
    )
    output = effect.apply(wav[:, None], sr).squeeze(1)
    return F.pad(output[: wav.numel()], (0, max(0, wav.numel() - output.numel())))


def _packet_loss(wav: torch.Tensor, sr: int) -> torch.Tensor:
    output = wav.clone()
    target = int(0.09 * wav.numel())
    removed = 0
    while removed < target and wav.numel() >= int(0.04 * sr):
        size = random.randint(int(0.02 * sr), min(int(0.2 * sr), wav.numel() // 2))
        start = random.randint(0, wav.numel() - size)
        output[start : start + size] = 0
        removed += size
    return output


# Degradations that have already failed once, so the warning below is emitted
# a single time per process rather than on every utterance.
_DEGRADE_WARNED: set = set()


def degrade_waveform(
    wav: torch.Tensor,
    sr: int,
    noise_files: List[str],
    rir_files: List[str],
    probability: float = 0.5,
) -> torch.Tensor:
    """Apply the six independent degradations described by Sidon."""
    original = wav.float()
    output = original.clone()
    operations = (
        lambda x: _reverb(x, sr, rir_files),
        lambda x: _noise(x, sr, noise_files),
        lambda x: _band_limit(x, sr),
        _clip,
        lambda x: _codec(x, sr),
        lambda x: _packet_loss(x, sr),
    )
    names = ("reverb", "noise", "band_limit", "clip", "codec", "packet_loss")
    for name, operation in zip(names, operations):
        if random.random() < probability:
            try:
                output = operation(output)
            except Exception as error:
                # A degradation that raises on every call silently removes
                # itself from the training distribution, and at debug level
                # nobody finds out. Two of the six were disabled this way:
                # codec raises whenever torchaudio's FFmpeg extension is
                # unavailable, and reverb was a no-op because the RIR pool
                # had been generated as unit impulses. Warn once per
                # degradation per process -- loud enough to notice in a log,
                # quiet enough not to flood it.
                if name not in _DEGRADE_WARNED:
                    _DEGRADE_WARNED.add(name)
                    logger.warning(
                        "Sidon degradation %r failed and is being SKIPPED for "
                        "every utterance in this process: %s: %s. The "
                        "degradation distribution is now missing this "
                        "component.",
                        name,
                        type(error).__name__,
                        error,
                    )
    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
    return original.clamp(-1, 1) if output.abs().max() < 1e-8 else output


class SidonCollateFn:
    """Collate function for Sidon: online degradation + padding.

    SSL feature extraction is done on GPU in the model forward pass
    (W2VBert2Encoder._wav_to_ssl_inputs), not here.
    """

    def __init__(
        self,
        max_samples: int,
        input_sr: int,
        noise_dir: str,
        rir_dir: str,
        degrade_prob: float,
        online_degradation: bool,
        train: bool = True,
    ):
        self.max_samples = max_samples
        self.input_sr = input_sr
        self.train = train
        self.noise_files = _audio_files(noise_dir)
        self.rir_files = _audio_files(rir_dir)
        self.degrade_prob = degrade_prob
        self.online_degradation = online_degradation
        self.base = CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

    def __call__(self, data):
        processed = []
        for key, values in data:
            values = dict(values)
            clean = torch.as_tensor(values["speech_ref1"]).float()
            # Validation must be reproducible across runs and workers.
            # CPython randomises str hashing per process unless PYTHONHASHSEED
            # is set, so the builtin hash of the utterance id is not a stable
            # seed; crc32 is. The state is restored only after the crop below,
            # because the crop offset must be deterministic in validation too.
            rng_state = None
            if not self.train:
                rng_state = random.getstate()
                random.seed(zlib.crc32(key.encode("utf-8")))
            if self.online_degradation:
                noisy = degrade_waveform(
                    clean,
                    self.input_sr,
                    self.noise_files,
                    self.rir_files,
                    self.degrade_prob,
                )
            else:
                noisy = torch.as_tensor(values.get("noisy_speech", clean)).float()
            length = min(clean.numel(), noisy.numel())
            if length > self.max_samples:
                start = random.randint(0, length - self.max_samples)
                length = self.max_samples
            else:
                start = 0
            if rng_state is not None:
                random.setstate(rng_state)
            values["speech_ref1"] = clean[start : start + length].numpy()
            values["noisy_speech"] = noisy[start : start + length].numpy()
            processed.append((key, values))

        return self.base(processed)


class RestorationTask(AbsTask):
    num_optimizers = 1
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser):
        group = parser.add_argument_group("ESPnet-Sidon")
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default={"extract_feats_in_collect_stats": False},
        )
        group.add_argument(
            "--ssl_encoder",
            choices=sorted(SSL_ENCODERS),
            default="w2v_bert2",
            help="w2v_bert2: w2v-BERT 2.0 layer 8 (Sidon paper); "
            "xeus: XEUS block 10 via ESPnet SSLTask",
        )
        group.add_argument(
            "--ssl_encoder_conf",
            action=NestedDictAction,
            default={},
            help="Encoder keyword arguments, e.g. model_tag, target_layer",
        )
        group.add_argument("--lora_rank", type=int, default=64)
        group.add_argument("--lora_alpha", type=int, default=16)
        group.add_argument("--lora_dropout", type=float, default=0.1)
        group.add_argument("--input_sr", type=int, default=16000)
        group.add_argument("--max_duration", type=float, default=20.0)
        group.add_argument("--noise_dir", default="data/noise_pool")
        group.add_argument("--rir_dir", default="data/rir_pool")
        group.add_argument("--degrade_prob", type=float, default=0.5)
        group.add_argument("--online_degradation", type=str2bool, default=True)

    @classmethod
    def build_collate_fn(cls, args, train):
        return SidonCollateFn(
            max_samples=int(args.max_duration * args.input_sr),
            input_sr=args.input_sr,
            noise_dir=args.noise_dir,
            rir_dir=args.rir_dir,
            degrade_prob=args.degrade_prob,
            online_degradation=args.online_degradation,
            train=train,
        )

    @classmethod
    def build_preprocess_fn(cls, args, train):
        return None

    @classmethod
    def required_data_names(cls, train=True, inference=False):
        return ("noisy_speech",) if inference else ("speech_ref1",)

    @classmethod
    def optional_data_names(cls, train=True, inference=False):
        return ("noisy_speech",)

    @classmethod
    def build_model(cls, args):
        encoder = build_ssl_encoder(
            args.ssl_encoder,
            args.ssl_encoder_conf,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            input_sr=args.input_sr,
        )
        return SidonFeaturePredictor(encoder)

    @classmethod
    def get_trainer(cls):
        return Trainer
