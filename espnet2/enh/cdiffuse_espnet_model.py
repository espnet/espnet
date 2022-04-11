"""Enhancement model module."""
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import OrderedDict
from typing import Optional
from typing import Tuple
import numpy as np
import logging

import torch
from typeguard import check_argument_types

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.enh.espnet_model import (
    ESPnetEnhancementModel as ESPnetEnhancementModelSupervised,
)


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")

EPS = torch.finfo(torch.get_default_dtype()).eps


class ESPnetEnhancementModel(ESPnetEnhancementModelSupervised, AbsESPnetModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self,
        encoder: AbsEncoder,
        separator: AbsSeparator,
        decoder: AbsDecoder,
        loss_wrappers: List[AbsLossWrapper],
        mask_type: Optional[str] = None,
        n_fft: int = 1024,
        hop_length: int = 256,
        mix_noisy: float = 0.2,
        noise_schedule: list = np.linspace(1e-4, 0.035, 50).tolist(),
        inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.35],
        fast_sampling=True,
        **kwargs,
    ):
        assert check_argument_types()

        super().__init__(
            encoder=encoder,
            separator=separator,
            decoder=decoder,
            loss_wrappers=loss_wrappers,
            mask_type=mask_type,
        )
        # Conditioner
        self.hop_length = hop_length
        self.conditioner = STFTEncoder(
            n_fft=n_fft, win_length=n_fft, hop_length=self.hop_length
        )

        beta = np.array(noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        if fast_sampling:
            self.inference_schedule(noise_schedule, inference_noise_schedule)
        else:
            self.inference_schedule(noise_schedule, noise_schedule)
        self.mix_noisy = mix_noisy

    def inference_schedule(self, train_noise_schedule, inference_noise_schedule):
        training_noise_schedule = np.array(train_noise_schedule)
        inference_noise_schedule = np.array(inference_noise_schedule)
        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        self.T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                        talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5
                    )
                    self.T.append(t + twiddle)
                    break
        self.T = np.array(self.T, dtype=np.float32)

        m = [0 for i in alpha]
        delta = [0 for i in alpha]
        d_x = [0 for i in alpha]
        d_y = [0 for i in alpha]
        delta_cond = [0 for i in alpha]
        self.delta_bar = [0 for i in alpha]
        self.c1 = [0 for i in alpha]
        self.c2 = [0 for i in alpha]
        self.c3 = [0 for i in alpha]

        for n in range(len(alpha)):
            m[n] = min(((1 - alpha_cum[n]) / (alpha_cum[n] ** 0.5)), 1) ** 0.5
            m[-1] = 1

        for n in range(len(alpha)):
            delta[n] = max(1 - (1 + m[n] ** 2) * alpha_cum[n], 0)

        for n in range(len(alpha)):
            if n > 0:
                d_x[n] = (1 - m[n]) / (1 - m[n - 1]) * (alpha[n] ** 0.5)
                d_y[n] = (m[n] - (1 - m[n]) / (1 - m[n - 1]) * m[n - 1]) * (
                    alpha_cum[n] ** 0.5
                )
                delta_cond[n] = (
                    delta[n]
                    - (((1 - m[n]) / (1 - m[n - 1]))) ** 2 * alpha[n] * delta[n - 1]
                )
                self.delta_bar[n] = (delta_cond[n]) * delta[n - 1] / delta[n]
            else:
                d_x[n] = (1 - m[n]) * (alpha[n] ** 0.5)
                d_y[n] = (m[n]) * (alpha_cum[n] ** 0.5)
                delta_cond[n] = 0
                self.delta_bar[n] = 0

        logging.info("beta: {}".format(beta))
        logging.info("alpha_cum: {}".format(" ".join(map(str, talpha_cum))))
        logging.info("gamma_cum: {}".format(" ".join(map(str, alpha_cum))))
        logging.info("m: {}".format(" ".join(map(str, m))))
        logging.info("delta: {}".format(" ".join(map(str, delta))))
        logging.info("d_x: {}".format(" ".join(map(str, d_x))))
        logging.info("d_y: {}".format(" ".join(map(str, d_y))))
        logging.info("delta_cond: {}".format(" ".join(map(str, delta_cond))))
        logging.info("self.delta_bar: {}".format(" ".join(map(str, self.delta_bar))))

        for n in range(len(alpha)):
            if n > 0:
                self.c1[n] = (1 - m[n]) / (1 - m[n - 1]) * (
                    delta[n - 1] / delta[n]
                ) * alpha[n] ** 0.5 + (1 - m[n - 1]) * (
                    delta_cond[n] / delta[n]
                ) / alpha[
                    n
                ] ** 0.5
                self.c2[n] = (
                    m[n - 1] * delta[n]
                    - (m[n] * (1 - m[n])) / (1 - m[n - 1]) * alpha[n] * delta[n - 1]
                ) * (alpha_cum[n - 1] ** 0.5 / delta[n])
                self.c3[n] = (
                    (1 - m[n - 1])
                    * (delta_cond[n] / delta[n])
                    * (1 - alpha_cum[n]) ** 0.5
                    / (alpha[n]) ** 0.5
                )
            else:
                self.c1[n] = 1 / alpha[n] ** 0.5
                self.c3[n] = self.c1[n] * beta[n] / (1 - alpha_cum[n]) ** 0.5

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
        """
        # clean speech signal of each speaker
        speech_ref = [
            kwargs["speech_ref{}".format(spk + 1)] for spk in range(self.num_spk)
        ]
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)

        if "noise_ref1" in kwargs:
            # noise signal (optional, required when using
            # frontend models with beamformering)
            noise_ref = [
                kwargs["noise_ref{}".format(n + 1)] for n in range(self.num_noise_type)
            ]
            # (Batch, num_noise_type, samples) or
            # (Batch, num_noise_type, samples, channels)
            noise_ref = torch.stack(noise_ref, dim=1)
        else:
            noise_ref = None

        # dereverberated (noisy) signal
        # (optional, only used for frontend models with WPE)
        if "dereverb_ref1" in kwargs:
            # noise signal (optional, required when using
            # frontend models with beamformering)
            dereverb_speech_ref = [
                kwargs["dereverb_ref{}".format(n + 1)]
                for n in range(self.num_spk)
                if "dereverb_ref{}".format(n + 1) in kwargs
            ]
            assert len(dereverb_speech_ref) in (1, self.num_spk), len(
                dereverb_speech_ref
            )
            # (Batch, N, samples) or (Batch, N, samples, channels)
            dereverb_speech_ref = torch.stack(dereverb_speech_ref, dim=1)
        else:
            dereverb_speech_ref = None

        batch_size = speech_mix.shape[0]
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int().fill_(speech_mix.shape[1])
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )

        # for data-parallel
        speech_ref = speech_ref[..., : speech_lengths.max()]
        speech_ref = speech_ref.unbind(dim=1)

        speech_mix = speech_mix[:, : speech_lengths.max()]

        # TODO(xkc09): Diffusion Process
        (
            speech_pre,
            feature_noisy,
            feature_pre,
            others,
            combine_noise,
            speech_lengths,
        ) = self._diffusion_process(speech_mix, speech_ref[0], speech_lengths)

        loss, stats, weight = self.forward_loss(
            speech_pre,
            speech_lengths,
            feature_noisy,
            feature_pre,
            others,
            combine_noise,
            noise_ref,
            dereverb_speech_ref,
        )

        return loss, stats, weight

    def _diffusion_process(self, speech_mix, speech_ref, speech_lengths):
        spectrogram, _ = self.conditioner(speech_mix, speech_lengths)
        spectrogram = torch.transpose(spectrogram.abs(), 1, 2)[:, :, :-1]

        # cut the audio files
        speech_lengths = [
            self.hop_length * spectrogram.shape[-1] for i in speech_lengths
        ]
        speech_mix = speech_mix[:, : speech_lengths[0]]
        speech_ref = speech_ref[:, : speech_lengths[0]]

        # diffusion process
        t = torch.randint(
            0, len(self.noise_level), [len(speech_lengths)], device=speech_ref.device
        )
        noise_scale = self.noise_level[t].unsqueeze(1).to(speech_ref.device)
        noise_scale_sqrt = noise_scale**0.5
        m = (
            (((1 - self.noise_level[t]) / self.noise_level[t] ** 0.5) ** 0.5)
            .unsqueeze(1)
            .to(speech_ref.device)
        )
        noise = torch.randn_like(speech_ref)

        noisy_audio = (
            (1 - m) * noise_scale_sqrt * speech_ref
            + m * noise_scale_sqrt * speech_mix
            + (1.0 - (1 + m**2) * noise_scale) ** 0.5 * noise
        )
        combine_noise = (
            m * noise_scale_sqrt * (speech_mix - speech_ref)
            + (1.0 - (1 + m**2) * noise_scale) ** 0.5 * noise
        ) / (1 - noise_scale) ** 0.5

        # Null encoder/decoder
        feature_noisy, flens = self.encoder(noisy_audio, speech_lengths)
        feature_pre, flens, others = self.separator(
            feature_noisy, spectrogram, t, flens
        )

        if feature_pre is not None:
            speech_pre = [self.decoder(ps, speech_lengths)[0] for ps in feature_pre]
        else:
            # some models (e.g. neural beamformer trained with mask loss)
            # do not predict time-domain signal in the training stage
            speech_pre = None

        return (
            speech_pre,
            feature_noisy,
            feature_pre,
            others,
            [combine_noise],
            speech_lengths,
        )

    def _reverse_process(self, speech_mix, speech_lengths):
        spectrogram, _ = self.conditioner(speech_mix, speech_lengths)
        spectrogram = torch.transpose(spectrogram.abs(), 1, 2)[:, :, :]

        audio = torch.randn(
            spectrogram.shape[0],
            self.hop_length * spectrogram.shape[-1],
            device=speech_mix.device,
        )
        noisy_audio = torch.zeros(
            spectrogram.shape[0],
            self.hop_length * spectrogram.shape[-1],
            device=speech_mix.device,
        )
        noisy_audio[:, : speech_mix.shape[1]] = speech_mix
        audio = noisy_audio
        for n in range(len(self.delta_bar) - 1, -1, -1):
            if n > 0:
                predicted_noise, _, _ = self.separator(
                    audio,
                    spectrogram,
                    torch.tensor([self.T[n]], device=speech_mix.device),
                    speech_lengths,
                )
                audio = (
                    self.c1[n] * audio
                    + self.c2[n] * noisy_audio
                    - self.c3[n] * predicted_noise.squeeze(1)
                )
                noise = torch.randn_like(audio)
                newsigma = self.delta_bar[n] ** 0.5
                audio += newsigma * noise
            else:
                predicted_noise, _, _ = self.separator(
                    audio,
                    spectrogram,
                    torch.tensor([self.T[n]], device=speech_mix.device),
                    speech_lengths,
                )
                audio = self.c1[n] * audio - self.c3[n] * predicted_noise.squeeze(1)
                audio = (1 - self.mix_noisy) * audio + self.mix_noisy * noisy_audio
            audio = torch.clamp(audio, -1.0, 1.0)

        return [audio[:, : speech_mix.shape[1]]], speech_lengths, None

    def forward_enhance(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # reverse process
        feature_mix, flens = self.encoder(speech_mix, speech_lengths)

        feature_pre, flens, others = self._reverse_process(feature_mix, flens)

        if feature_pre is not None:
            speech_pre = [self.decoder(ps, speech_lengths)[0] for ps in feature_pre]
        else:
            # some models (e.g. neural beamformer trained with mask loss)
            # do not predict time-domain signal in the training stage
            speech_pre = None
        return speech_pre, feature_mix, feature_pre, others

    # forward_loss can be removed after the enh_asr branch is merged
    def forward_loss(
        self,
        speech_pre: torch.Tensor,
        speech_lengths: torch.Tensor,
        feature_mix: torch.Tensor,
        feature_pre: torch.Tensor,
        others: OrderedDict,
        speech_ref: torch.Tensor,
        noise_ref: torch.Tensor = None,
        dereverb_speech_ref: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        loss = 0.0
        stats = dict()
        o = {}
        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            if isinstance(criterion, TimeDomainLoss):
                if speech_ref[0].dim() == 3:
                    # For multi-channel reference,
                    # only select one channel as the reference
                    speech_ref = [sr[..., self.ref_channel] for sr in speech_ref]
                # for the time domain criterions
                l, s, o = loss_wrapper(speech_ref, speech_pre, o)
            elif isinstance(criterion, FrequencyDomainLoss):
                # for the time-frequency domain criterions
                if criterion.compute_on_mask:
                    # compute on mask
                    tf_ref = criterion.create_mask_label(
                        feature_mix,
                        [self.encoder(sr, speech_lengths)[0] for sr in speech_ref],
                    )
                    tf_pre = [
                        others["mask_spk{}".format(spk + 1)]
                        for spk in range(self.num_spk)
                    ]
                else:
                    # compute on spectrum
                    if speech_ref[0].dim() == 3:
                        # For multi-channel reference,
                        # only select one channel as the reference
                        speech_ref = [sr[..., self.ref_channel] for sr in speech_ref]
                    tf_ref = [self.encoder(sr, speech_lengths)[0] for sr in speech_ref]
                    tf_pre = feature_pre

                l, s, o = loss_wrapper(tf_ref, tf_pre, o)
            loss += l * loss_wrapper.weight
            stats.update(s)

        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        batch_size = speech_ref[0].shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
