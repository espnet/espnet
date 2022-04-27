"""Enhancement model module."""
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import OrderedDict
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")

EPS = torch.finfo(torch.get_default_dtype()).eps


class ESPnetEnhancementModel(AbsESPnetModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self,
        encoder: AbsEncoder,
        separator: AbsSeparator,
        decoder: AbsDecoder,
        loss_wrappers: List[AbsLossWrapper],
        stft_consistency: bool = False,
        loss_type: str = "mask_mse",
        mask_type: Optional[str] = None,
    ):
        assert check_argument_types()

        super().__init__()

        self.encoder = encoder
        self.separator = separator
        self.decoder = decoder
        self.loss_wrappers = loss_wrappers
        self.num_spk = separator.num_spk
        self.num_noise_type = getattr(self.separator, "num_noise_type", 1)

        # get mask type for TF-domain models
        # (only used when loss_type="mask_*") (deprecated, keep for compatibility)
        self.mask_type = mask_type.upper() if mask_type else None

        # get loss type for model training (deprecated, keep for compatibility)
        self.loss_type = loss_type

        # whether to compute the TF-domain loss
        # while enforcing STFT consistency (deprecated, keep for compatibility)
        self.stft_consistency = stft_consistency

        # for multi-channel signal
        self.ref_channel = getattr(self.separator, "ref_channel", -1)

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
            kwargs: "utt_id" is among the input.
        """
        # clean speech signal of each speaker
        speech_ref = [
            kwargs["speech_ref{}".format(spk + 1)] for spk in range(self.num_spk)
        ]
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)

        if "noise_ref1" in kwargs:
            # noise signal (optional, required when using beamforming-based
            # frontend models)
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
        additional = {}
        additional["feature_ref"] = [
            self.encoder(r, speech_lengths)[0] for r in speech_ref
        ]

        speech_mix = speech_mix[:, : speech_lengths.max()]

        # model forward
        speech_pre, feature_mix, feature_pre, others = self.forward_enhance(
            speech_mix, speech_lengths, additional
        )

        # loss computation
        loss, stats, weight = self.forward_loss(
            speech_pre,
            speech_lengths,
            feature_mix,
            feature_pre,
            others,
            speech_ref,
            noise_ref,
            dereverb_speech_ref,
        )
        return loss, stats, weight

    def forward_enhance(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_mix, flens = self.encoder(speech_mix, speech_lengths)
        feature_pre, flens, others = self.separator(feature_mix, flens, additional)
        if feature_pre is not None:
            speech_pre = [self.decoder(ps, speech_lengths)[0] for ps in feature_pre]
        else:
            # some models (e.g. neural beamformer trained with mask loss)
            # do not predict time-domain signal in the training stage
            speech_pre = None
        return speech_pre, feature_mix, feature_pre, others

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
                l, s, o = loss_wrapper(speech_ref, speech_pre, others)
            elif isinstance(criterion, FrequencyDomainLoss):
                # for the time-frequency domain criterions
                if criterion.compute_on_mask:
                    # compute loss on masks
                    if noise_ref is not None:
                        noise_spec = self.encoder(noise_ref.sum(1), speech_lengths)[0]
                    else:
                        noise_spec = None
                    tf_ref = criterion.create_mask_label(
                        feature_mix,
                        [self.encoder(sr, speech_lengths)[0] for sr in speech_ref],
                        noise_spec=noise_spec,
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

                l, s, o = loss_wrapper(tf_ref, tf_pre, others)
            else:
                raise NotImplementedError("Unsupported loss type: %s" % str(criterion))

            loss += l * loss_wrapper.weight
            stats.update(s)

        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
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
