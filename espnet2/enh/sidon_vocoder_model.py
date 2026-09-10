"""GAN training of the Sidon vocoder (recipe stages 2 and 3).

Stage 2 pretrains the vocoder to invert *ground-truth* w2v-BERT 2.0 layer-8
features of clean 48 kHz speech; stage 3 finetunes it on features *predicted*
by the frozen stage-1 predictor from degraded 16 kHz input, closing the gap
between what the vocoder saw in training and what it receives at test time.
Both stages use the same losses (mel L1, LSGAN adversarial, feature matching)
and the same DAC-style discriminator; the only difference is which encoder
produces the features.

Reuses ESPnet's own GAN infrastructure rather than carrying a parallel copy:
the multi-period + multi-band discriminator from ``espnet2.gan_codec`` and
the HiFi-GAN losses from ``espnet2.gan_tts``, both driven by ``GANTrainer``.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torchaudio.functional as AF
from torch import nn

from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
    MelSpectrogramLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel

logger = logging.getLogger(__name__)

# w2v-BERT 2.0 emits one feature vector per 20 ms regardless of sample rate.
SSL_FRAME_RATE = 50


class SidonVocoderFeatures:
    """Frozen-encoder feature extraction and excerpt cropping shared by the
    vocoder model (kept separate so other vocoder objectives can reuse
    it). Expects ``ssl_encoder``,
    ``use_predicted_feat``, ``input_sr``, ``output_sr``, ``hop``,
    ``segment_frames`` and ``segment_samples`` on the instance."""

    @torch.no_grad()
    def _ssl_features(
        self,
        speech_ref1: torch.Tensor,
        speech_ref1_lengths: torch.Tensor,
        noisy_speech: Optional[torch.Tensor],
        noisy_speech_lengths: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, T, D) features and per-utterance frame counts."""
        if self.use_predicted_feat:
            if noisy_speech is None:
                raise ValueError(
                    "use_predicted_feat=True needs noisy_speech in the batch; "
                    "the collate function produces it when online_degradation "
                    "is on or the data directory supplies it"
                )
            wav, lengths = noisy_speech, noisy_speech_lengths
        else:
            # The encoder only ever sees 16 kHz. Resampling the 48 kHz
            # reference on the GPU keeps the collate function light for
            # pretraining, which needs no degradation at all.
            wav = AF.resample(speech_ref1, self.output_sr, self.input_sr)
            lengths = speech_ref1_lengths * self.input_sr // self.output_sr
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=wav.is_cuda):
            ssl_inputs = self.ssl_encoder._wav_to_ssl_inputs(wav, lengths)
            feat, mask = self.ssl_encoder.encode(
                ssl_inputs, teacher=not self.use_predicted_feat
            )
        feat_lengths = mask.sum(dim=1).clamp(max=feat.size(1))
        return feat.float(), feat_lengths

    def _crop(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        speech_ref1: torch.Tensor,
        speech_ref1_lengths: torch.Tensor,
        crop_start: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aligned fixed-length excerpts: frames [s, s+F) and samples
        [s*hop, (s+F)*hop). Utterances shorter than the segment are
        zero-padded on both sides of the pair."""
        B, _, D = feat.shape
        feat_out = feat.new_zeros(B, self.segment_frames, D)
        wav_out = speech_ref1.new_zeros(B, self.segment_samples)
        for i in range(B):
            n_frames = int(feat_lengths[i].item())
            start = min(
                int(crop_start[i].item()), max(0, n_frames - self.segment_frames)
            )
            end = min(start + self.segment_frames, n_frames, feat.size(1))
            if end > start:
                feat_out[i, : end - start] = feat[i, start:end]
            w_start = start * self.hop
            w_end = min(
                w_start + self.segment_samples,
                int(speech_ref1_lengths[i].item()),
                speech_ref1.size(1),
            )
            if w_end > w_start:
                wav_out[i, : w_end - w_start] = speech_ref1[i, w_start:w_end]
        return feat_out, wav_out

    @staticmethod
    def _trim(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # The DAC decoder's odd strides (5, 3) leave the output a few samples
        # short of frames * 960; compare over the common length.
        n = min(a.size(-1), b.size(-1))
        return a[..., :n], b[..., :n]


class SidonVocoderGAN(SidonVocoderFeatures, AbsGANESPnetModel):
    """Vocoder generator + discriminator with a frozen feature encoder.

    Args:
        ssl_encoder: stage-1 encoder (W2VBert2Encoder or XeusEncoder). Always
            frozen here; the teacher branch provides ground-truth features
            (pretrain) and the LoRA student branch provides predicted ones
            (finetune).
        vocoder: generator mapping (B, T, D) features to (B, T * 960) audio
            (SidonVocoder or SidonHiFiGANVocoder; anything with ``generate``
            and ``upsample_factor``).
        discriminator: returns a list, one entry per sub-discriminator, of
            [feature maps ..., logits].
        use_predicted_feat: False for stage 2 (teacher on clean speech),
            True for stage 3 (student on degraded speech).
        input_sr: encoder sample rate (16 kHz).
        output_sr: vocoder sample rate (48 kHz).
        segment_duration: length in seconds of the aligned feature/waveform
            excerpt the generator and discriminator actually train on. The
            encoder runs on the longer context the collate function provides,
            so the excerpt's features are computed with the surrounding
            speech in view, as they are at inference.
        mel_loss_weight, adv_loss_weight, fm_loss_weight: generator loss
            weights (Sidon: 15 / 2 / 1).
        mel_loss_conf: overrides for MelSpectrogramLoss.
    """

    def __init__(
        self,
        ssl_encoder: nn.Module,
        vocoder: nn.Module,
        discriminator: nn.Module,
        use_predicted_feat: bool = False,
        input_sr: int = 16000,
        output_sr: int = 48000,
        segment_duration: float = 1.0,
        mel_loss_weight: float = 15.0,
        adv_loss_weight: float = 2.0,
        fm_loss_weight: float = 1.0,
        mel_loss_conf: Optional[Dict] = None,
    ):
        super().__init__()
        if output_sr % SSL_FRAME_RATE != 0:
            raise ValueError(f"output_sr must be a multiple of {SSL_FRAME_RATE} Hz")
        self.hop = output_sr // SSL_FRAME_RATE
        if vocoder.upsample_factor != self.hop:
            raise ValueError(
                f"vocoder upsamples {vocoder.upsample_factor}x but {output_sr} Hz "
                f"output needs {self.hop}x per {SSL_FRAME_RATE} Hz frame"
            )
        self.ssl_encoder = ssl_encoder
        self.ssl_encoder.requires_grad_(False)
        self.vocoder = vocoder
        self.discriminator = discriminator
        self.use_predicted_feat = use_predicted_feat
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.segment_frames = max(1, int(round(segment_duration * SSL_FRAME_RATE)))
        self.segment_samples = self.segment_frames * self.hop
        self.mel_loss_weight = mel_loss_weight
        self.adv_loss_weight = adv_loss_weight
        self.fm_loss_weight = fm_loss_weight

        mel_conf = dict(
            fs=output_sr,
            n_fft=2048,
            hop_length=480,
            win_length=2048,
            n_mels=128,
            fmin=0,
            fmax=None,
            log_base=10.0,
        )
        mel_conf.update(mel_loss_conf or {})
        self.mel_loss = MelSpectrogramLoss(**mel_conf)
        # Sidon inherits DAC's GAN loss, which sums over sub-discriminators
        # and feature-map layers rather than averaging; with eight
        # sub-discriminators the averaged form would make the adversarial and
        # feature-matching terms eight times weaker relative to the mel term
        # than the published weights intend.
        self.generator_adv_loss = GeneratorAdversarialLoss(
            average_by_discriminators=False, loss_type="mse"
        )
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            average_by_discriminators=False, loss_type="mse"
        )
        self.feat_match_loss = FeatureMatchLoss(
            average_by_layers=False,
            average_by_discriminators=False,
            include_final_outputs=False,
        )

    # ------------------------------------------------------------------
    # features
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # GAN turns
    # ------------------------------------------------------------------
    def forward(
        self,
        speech_ref1: torch.Tensor,
        speech_ref1_lengths: torch.Tensor,
        noisy_speech: Optional[torch.Tensor] = None,
        noisy_speech_lengths: Optional[torch.Tensor] = None,
        vocoder_crop_start: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
        **kwargs,
    ):
        if vocoder_crop_start is None:
            vocoder_crop_start = speech_ref1.new_zeros(
                speech_ref1.size(0), dtype=torch.long
            )
        feat, feat_lengths = self._ssl_features(
            speech_ref1, speech_ref1_lengths, noisy_speech, noisy_speech_lengths
        )
        feat, real = self._crop(
            feat, feat_lengths, speech_ref1, speech_ref1_lengths, vocoder_crop_start
        )
        if forward_generator:
            return self._forward_generator(feat, real)
        return self._forward_discriminator(feat, real)

    def _forward_generator(self, feat: torch.Tensor, real: torch.Tensor):
        fake = self.vocoder.generate(feat)
        fake, real = self._trim(fake, real)
        fake, real = fake.unsqueeze(1), real.unsqueeze(1)

        mel = self.mel_loss(fake, real)
        d_fake = self.discriminator(fake)
        with torch.no_grad():
            d_real = self.discriminator(real)
        adv = self.generator_adv_loss(d_fake)
        fm = self.feat_match_loss(d_fake, d_real)
        loss = (
            self.mel_loss_weight * mel
            + self.adv_loss_weight * adv
            + self.fm_loss_weight * fm
        )
        stats = dict(
            loss_G=loss.detach(),
            loss_mel=mel.detach(),
            loss_adv_G=adv.detach(),
            loss_fm=fm.detach(),
        )
        loss, stats, weight = force_gatherable((loss, stats, feat.size(0)), loss.device)
        return dict(loss=loss, stats=stats, weight=weight, optim_idx=0)

    def _forward_discriminator(self, feat: torch.Tensor, real: torch.Tensor):
        with torch.no_grad():
            fake = self.vocoder.generate(feat)
        fake, real = self._trim(fake, real)
        d_fake = self.discriminator(fake.unsqueeze(1))
        d_real = self.discriminator(real.unsqueeze(1))
        real_loss, fake_loss = self.discriminator_adv_loss(d_fake, d_real)
        loss = real_loss + fake_loss
        stats = dict(
            loss_D=loss.detach(),
            loss_D_real=real_loss.detach(),
            loss_D_fake=fake_loss.detach(),
        )
        loss, stats, weight = force_gatherable((loss, stats, feat.size(0)), loss.device)
        return dict(loss=loss, stats=stats, weight=weight, optim_idx=1)

    def collect_feats(self, **batch):
        return {}

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen in both stages: no dropout, no LoRA dropout, no BN updates.
        self.ssl_encoder.eval()
        return self
