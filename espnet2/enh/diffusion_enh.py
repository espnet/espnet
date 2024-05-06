"""Enhancement model module."""

from typing import Dict, Tuple

import torch
from typeguard import typechecked

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.diffusion.abs_diffusion import AbsDiffusion
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.extractor.abs_extractor import AbsExtractor  # noqa
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss  # noqa
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss  # noqa
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper  # noqa
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel  # noqa

EPS = torch.finfo(torch.get_default_dtype()).eps


class ESPnetDiffusionModel(ESPnetEnhancementModel):
    """Target Speaker Extraction Frontend model"""

    @typechecked
    def __init__(
        self,
        encoder: AbsEncoder,
        diffusion: AbsDiffusion,
        decoder: AbsDecoder,
        # loss_wrappers: List[AbsLossWrapper],
        num_spk: int = 1,
        normalize: bool = False,
        **kwargs,
    ):

        super().__init__(
            encoder=encoder,
            separator=None,
            decoder=decoder,
            mask_module=None,
            loss_wrappers=None,
            **kwargs,
        )

        self.encoder = encoder
        self.diffusion = diffusion
        self.decoder = decoder

        # TODO(gituser): Extending the model to separation tasks.
        assert (
            num_spk == 1
        ), "only enhancement models are supported now, num_spk must be 1"
        self.num_spk = num_spk
        self.normalize = normalize

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref1: (Batch, samples)
                        or (Batch, samples, channels)
            speech_ref2: (Batch, samples)
                        or (Batch, samples, channels)
            ...
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            enroll_ref1: (Batch, samples_aux)
                                enrollment (raw audio or embedding) for speaker 1
            enroll_ref2: (Batch, samples_aux)
                                enrollment (raw audio or embedding) for speaker 2
            ...
            kwargs: "utt_id" is among the input.
        """
        # reference speech signal of each speaker
        assert "speech_ref1" in kwargs, "At least 1 reference signal input is required."
        speech_ref = [
            kwargs.get(
                f"speech_ref{spk + 1}",
                torch.zeros_like(kwargs["speech_ref1"]),
            )
            for spk in range(self.num_spk)
            if "speech_ref{}".format(spk + 1) in kwargs
        ]
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)
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
        speech_ref = speech_ref[..., : speech_lengths.max()].unbind(dim=1)
        speech_mix = speech_mix[:, : speech_lengths.max()]

        if self.normalize:
            normfac = speech_mix.abs().max() * 1.1 + 1e-5
        else:
            normfac = 1.0

        speech_mix = speech_mix / normfac
        speech_ref = [r / normfac for r in speech_ref]

        # loss computation
        loss, stats, weight = self.forward_loss(
            speech_ref=speech_ref, speech_mix=speech_mix, speech_lengths=speech_lengths
        )
        return loss, stats, weight

    def enhance(self, feature_mix):
        if self.normalize:
            normfac = feature_mix.abs().max() * 1.1 + 1e-5
            feature_mix = feature_mix / normfac

        return self.diffusion.enhance(feature_mix)

    def forward_loss(
        self,
        speech_ref,
        speech_mix,
        speech_lengths,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        feature_mix, flens = self.encoder(speech_mix, speech_lengths)
        feature_ref, flens = self.encoder(speech_ref[0], speech_lengths)

        stats = {}
        loss = self.diffusion(feature_ref=feature_ref, feature_mix=feature_mix)
        stats["loss"] = loss.detach()
        batch_size = speech_ref[0].shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
