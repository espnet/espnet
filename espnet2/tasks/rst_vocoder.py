"""Task definition for training the restoration vocoder (Sidon stages 2 and 3)."""

import logging
import random
import zlib

import torch
import torchaudio.functional as AF
from torch import nn

from espnet2.gan_codec.shared.discriminator.msmpmb_discriminator import (
    MultiScaleMultiPeriodMultiBandDiscriminator,
)
from espnet2.rst.decoder.dac_vocoder import VOCODERS, build_vocoder
from espnet2.rst.rst_model import SSL_ENCODERS, build_ssl_encoder
from espnet2.rst.rst_vocoder_model import SSL_FRAME_RATE, ESPnetRestorationVocoderModel
from espnet2.tasks.abs_task import AbsTask, optim_classes
from espnet2.tasks.rst import _audio_files, degrade_waveform
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.gan_trainer import GANTrainer
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool, str_or_none

logger = logging.getLogger(__name__)


class RestorationVocoderCollateFn:
    """Cut a context window from the 48 kHz reference and pick the excerpt.

    Emits ``speech_ref1`` (48 kHz context), ``vocoder_crop_start`` (the SSL
    frame at which the model cuts its fixed-length training excerpt) and, for
    finetuning only, ``noisy_speech`` (16 kHz degraded context, the input to
    the frozen feature predictor). Pretraining needs no 16 kHz copy here: the
    model resamples on the GPU.

    With ``stats_only`` the reference passes through untouched so the shape
    file collected in stage 6 records true utterance lengths.
    """

    def __init__(
        self,
        context_samples: int,
        segment_frames: int,
        hop: int,
        input_sr: int,
        output_sr: int,
        use_predicted_feat: bool,
        noise_dir: str,
        rir_dir: str,
        degrade_prob: float,
        online_degradation: bool,
        train: bool = True,
        stats_only: bool = False,
    ):
        self.context_samples = context_samples
        self.segment_frames = segment_frames
        self.hop = hop
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.use_predicted_feat = use_predicted_feat
        self.degrade_prob = degrade_prob
        self.online_degradation = online_degradation
        self.train = train
        self.stats_only = stats_only
        self.noise_files = _audio_files(noise_dir) if use_predicted_feat else []
        self.rir_files = _audio_files(rir_dir) if use_predicted_feat else []
        self.base = CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

    def __call__(self, data):
        if self.stats_only:
            return self.base(data)
        processed = []
        crop_starts = []
        for key, values in data:
            values = dict(values)
            clean = torch.as_tensor(values["speech_ref1"]).float()
            # Validation excerpts must be the same every epoch and on every
            # worker; crc32 of the utterance id is a stable seed where the
            # builtin hash is not (see RestorationCollateFn).
            rng_state = None
            if not self.train:
                rng_state = random.getstate()
                random.seed(zlib.crc32(key.encode("utf-8")))
            length = clean.numel()
            start = 0
            if 0 < self.context_samples < length:
                # Frame-aligned so a frame index means the same thing in the
                # context as in the full utterance.
                start = random.randint(0, (length - self.context_samples) // self.hop)
                start *= self.hop
                length = self.context_samples
            context = clean[start : start + length]
            n_frames = length // self.hop
            crop_starts.append(
                random.randint(0, max(0, n_frames - self.segment_frames))
            )
            values["speech_ref1"] = context.numpy()
            if self.use_predicted_feat:
                if self.online_degradation:
                    clean16 = AF.resample(context, self.output_sr, self.input_sr)
                    noisy = degrade_waveform(
                        clean16,
                        self.input_sr,
                        self.noise_files,
                        self.rir_files,
                        self.degrade_prob,
                    )
                else:
                    noisy = torch.as_tensor(values["noisy_speech"]).float()
                    s = start * self.input_sr // self.output_sr
                    n = length * self.input_sr // self.output_sr
                    noisy = noisy[s : s + n]
                values["noisy_speech"] = noisy.numpy()
            else:
                values.pop("noisy_speech", None)
            if rng_state is not None:
                random.setstate(rng_state)
            processed.append((key, values))
        keys, batch = self.base(processed)
        batch["vocoder_crop_start"] = torch.tensor(crop_starts, dtype=torch.long)
        return keys, batch


class RestorationVocoderTask(AbsTask):
    num_optimizers = 2
    trainer = GANTrainer

    @classmethod
    def add_task_arguments(cls, parser):
        group = parser.add_argument_group("ESPnet restoration vocoder")
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default={"extract_feats_in_collect_stats": False},
        )
        group.add_argument(
            "--ssl_encoder", choices=sorted(SSL_ENCODERS), default="w2v_bert2"
        )
        group.add_argument("--ssl_encoder_conf", action=NestedDictAction, default={})
        group.add_argument("--lora_rank", type=int, default=64)
        group.add_argument("--lora_alpha", type=int, default=16)
        group.add_argument("--lora_dropout", type=float, default=0.1)
        group.add_argument("--input_sr", type=int, default=16000)
        group.add_argument("--output_sr", type=int, default=48000)
        group.add_argument(
            "--use_predicted_feat",
            type=str2bool,
            default=False,
            help="False: pretrain on teacher features of clean speech (stage 2). "
            "True: finetune on the frozen stage-1 predictor's features of "
            "degraded speech (stage 3); requires --fp_model_path.",
        )
        group.add_argument(
            "--fp_model_path",
            type=str_or_none,
            default=None,
            help="Stage-1 feature-predictor checkpoint (valid.loss.best.pth)",
        )
        group.add_argument(
            "--vocoder_type",
            choices=sorted(VOCODERS),
            default="dac",
            help="dac: DAC decoder as in Sidon (default); "
            "hifigan: ESPnet HiFi-GAN generator",
        )
        group.add_argument("--vocoder_conf", action=NestedDictAction, default={})
        group.add_argument("--discriminator_conf", action=NestedDictAction, default={})
        group.add_argument("--mel_loss_conf", action=NestedDictAction, default={})
        group.add_argument("--mel_loss_weight", type=float, default=15.0)
        group.add_argument("--adv_loss_weight", type=float, default=2.0)
        group.add_argument("--fm_loss_weight", type=float, default=1.0)
        group.add_argument(
            "--context_duration",
            type=float,
            default=8.0,
            help="Seconds of speech the encoder sees per utterance (0: all)",
        )
        group.add_argument(
            "--segment_duration",
            type=float,
            default=1.0,
            help="Seconds of aligned features/audio the GAN trains on",
        )
        group.add_argument("--noise_dir", default="data/noise_pool")
        group.add_argument("--rir_dir", default="data/rir_pool")
        group.add_argument("--degrade_prob", type=float, default=0.5)
        group.add_argument("--online_degradation", type=str2bool, default=True)

    @classmethod
    def build_collate_fn(cls, args, train):
        hop = args.output_sr // SSL_FRAME_RATE
        return RestorationVocoderCollateFn(
            context_samples=int(args.context_duration * args.output_sr),
            segment_frames=max(1, int(round(args.segment_duration * SSL_FRAME_RATE))),
            hop=hop,
            input_sr=args.input_sr,
            output_sr=args.output_sr,
            use_predicted_feat=args.use_predicted_feat,
            noise_dir=args.noise_dir,
            rir_dir=args.rir_dir,
            degrade_prob=args.degrade_prob,
            online_degradation=args.online_degradation,
            train=train,
            stats_only=bool(getattr(args, "collect_stats", False)),
        )

    @classmethod
    def build_preprocess_fn(cls, args, train):
        return None

    @classmethod
    def required_data_names(cls, train=True, inference=False):
        return ("speech_ref1",)

    @classmethod
    def optional_data_names(cls, train=True, inference=False):
        return ("noisy_speech",)

    @staticmethod
    def _load_feature_predictor(encoder: nn.Module, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        prefix = "ssl_encoder."
        state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        # LoRA B is zero-initialised, so a student whose adapter failed to
        # load is indistinguishable from the base model at run time: it
        # trains, it just trains against the wrong features. Refuse instead.
        loaded_lora = [k for k in state if "lora_" in k]
        missing_lora = [k for k in missing if "lora_" in k]
        if not loaded_lora or missing_lora:
            raise RuntimeError(
                f"{path} does not hold the LoRA adapter this encoder expects "
                f"(loaded {len(loaded_lora)}, missing {len(missing_lora)}); "
                f"use the stage-1 checkpoint with matching lora_rank/lora_alpha"
            )
        if unexpected:
            logger.warning("ignored %d unexpected tensors in %s", len(unexpected), path)
        logger.info("loaded frozen feature predictor from %s", path)

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
        if args.fp_model_path:
            cls._load_feature_predictor(encoder, args.fp_model_path)
        elif args.use_predicted_feat:
            raise ValueError("--use_predicted_feat true requires --fp_model_path")

        vocoder = build_vocoder(args.vocoder_type, encoder.ssl_dim, args.vocoder_conf)

        disc_conf = dict(
            rates=[],
            fft_sizes=[2048, 1024, 512],
            periods=[2, 3, 5, 7, 11],
            sample_rate=args.output_sr,
            band_discriminator_params={
                "hop_factor": 0.25,
                "sample_rate": args.output_sr,
                "bands": [
                    (0.0, 0.1),
                    (0.1, 0.25),
                    (0.25, 0.5),
                    (0.5, 0.75),
                    (0.75, 1.0),
                ],
                "channel": 32,
            },
        )
        disc_conf.update(args.discriminator_conf or {})
        discriminator = MultiScaleMultiPeriodMultiBandDiscriminator(**disc_conf)

        return ESPnetRestorationVocoderModel(
            ssl_encoder=encoder,
            vocoder=vocoder,
            discriminator=discriminator,
            use_predicted_feat=args.use_predicted_feat,
            input_sr=args.input_sr,
            output_sr=args.output_sr,
            segment_duration=args.segment_duration,
            mel_loss_weight=args.mel_loss_weight,
            adv_loss_weight=args.adv_loss_weight,
            fm_loss_weight=args.fm_loss_weight,
            mel_loss_conf=dict(args.mel_loss_conf or {}),
        )

    @classmethod
    def build_optimizers(cls, args, model):
        optim_g = optim_classes.get(args.optim)
        if optim_g is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        optim_d = optim_classes.get(args.optim2)
        if optim_d is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim2}")
        return [
            optim_g(model.vocoder.parameters(), **args.optim_conf),
            optim_d(model.discriminator.parameters(), **args.optim2_conf),
        ]

    @classmethod
    def get_trainer(cls):
        return GANTrainer
