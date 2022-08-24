"""Enhancement model module."""
from typing import Dict, List, Optional, OrderedDict, Tuple

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.diar.layers.abs_mask import AbsMask
from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.separator.dan_separator import DANSeparator
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

EPS = torch.finfo(torch.get_default_dtype()).eps


class ESPnetEnhancementModel(AbsESPnetModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self,
        encoder: AbsEncoder,
        separator: AbsSeparator,
        decoder: AbsDecoder,
        mask_module: Optional[AbsMask],
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
        self.mask_module = mask_module
        self.num_spk = separator.num_spk
        self.num_noise_type = getattr(self.separator, "num_noise_type", 1)

        self.loss_wrappers = loss_wrappers
        names = [w.criterion.name for w in self.loss_wrappers]
        if len(set(names)) != len(names):
            raise ValueError("Duplicated loss names are not allowed: {}".format(names))

        # get mask type for TF-domain models
        # (only used when loss_type="mask_*") (deprecated, keep for compatibility)
        self.mask_type = mask_type.upper() if mask_type else None

        # get loss type for model training (deprecated, keep for compatibility)
        self.loss_type = loss_type

        # whether to compute the TF-domain loss while enforcing STFT consistency
        # (deprecated, keep for compatibility)
        # NOTE: STFT consistency is now always used for frequency-domain spectrum losses
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
        speech_ref = speech_ref[..., : speech_lengths.max()].unbind(dim=1)
        if noise_ref is not None:
            noise_ref = noise_ref[..., : speech_lengths.max()].unbind(dim=1)
        if dereverb_speech_ref is not None:
            dereverb_speech_ref = dereverb_speech_ref[..., : speech_lengths.max()]
            dereverb_speech_ref = dereverb_speech_ref.unbind(dim=1)

        additional = {}
        # Additional data is required in Deep Attractor Network
        if isinstance(self.separator, DANSeparator):
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
        if self.mask_module is None:
            feature_pre, flens, others = self.separator(feature_mix, flens, additional)
        else:
            # Obtain bottleneck_feats from separator.
            # This is used for the input of diarization module in "enh + diar" task
            bottleneck_feats, bottleneck_feats_lengths = self.separator(
                feature_mix, flens
            )
            if additional.get("num_spk") is not None:
                feature_pre, flens, others = self.mask_module(
                    feature_mix, flens, bottleneck_feats, additional["num_spk"]
                )
                others["bottleneck_feats"] = bottleneck_feats
                others["bottleneck_feats_lengths"] = bottleneck_feats_lengths
            else:
                feature_pre = None
                others = {
                    "bottleneck_feats": bottleneck_feats,
                    "bottleneck_feats_lengths": bottleneck_feats_lengths,
                }
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
        # for calculating loss on estimated noise signals
        if getattr(self.separator, "predict_noise", False):
            assert "noise1" in others, others.keys()
        if noise_ref is not None and "noise1" in others:
            for n in range(self.num_noise_type):
                key = "noise{}".format(n + 1)
                others[key] = self.decoder(others[key], speech_lengths)[0]
        # for calculating loss on dereverberated signals
        if getattr(self.separator, "predict_dereverb", False):
            assert "dereverb1" in others, others.keys()
        if dereverb_speech_ref is not None and "dereverb1" in others:
            for spk in range(self.num_spk):
                key = "dereverb{}".format(spk + 1)
                if key in others:
                    others[key] = self.decoder(others[key], speech_lengths)[0]

        loss = 0.0
        stats = {}
        o = {}
        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            if getattr(criterion, "only_for_test", False) and self.training:
                continue
            if getattr(criterion, "is_noise_loss", False):
                if noise_ref is None:
                    raise ValueError(
                        "No noise reference for training!\n"
                        'Please specify "--use_noise_ref true" in run.sh'
                    )
                signal_ref = noise_ref
                signal_pre = [
                    others["noise{}".format(n + 1)] for n in range(self.num_noise_type)
                ]
            elif getattr(criterion, "is_dereverb_loss", False):
                if dereverb_speech_ref is None:
                    raise ValueError(
                        "No dereverberated reference for training!\n"
                        'Please specify "--use_dereverb_ref true" in run.sh'
                    )
                signal_ref = dereverb_speech_ref
                signal_pre = [
                    others["dereverb{}".format(n + 1)]
                    for n in range(self.num_noise_type)
                    if "dereverb{}".format(n + 1) in others
                ]
                if len(signal_pre) == 0:
                    signal_pre = None
            else:
                signal_ref = speech_ref
                signal_pre = speech_pre

            if isinstance(criterion, TimeDomainLoss):
                assert signal_pre is not None
                sref, spre = self._align_ref_pre_channels(
                    signal_ref, signal_pre, ch_dim=2, force_1ch=True
                )
                # for the time domain criterions
                l, s, o = loss_wrapper(sref, spre, {**others, **o})
            elif isinstance(criterion, FrequencyDomainLoss):
                sref, spre = self._align_ref_pre_channels(
                    signal_ref, signal_pre, ch_dim=2, force_1ch=False
                )
                # for the time-frequency domain criterions
                if criterion.compute_on_mask:
                    # compute loss on masks
                    if getattr(criterion, "is_noise_loss", False):
                        tf_ref, tf_pre = self._get_noise_masks(
                            criterion,
                            feature_mix,
                            speech_ref,
                            signal_ref,
                            signal_pre,
                            speech_lengths,
                            others,
                        )
                    elif getattr(criterion, "is_dereverb_loss", False):
                        tf_ref, tf_pre = self._get_dereverb_masks(
                            criterion,
                            feature_mix,
                            noise_ref,
                            signal_ref,
                            signal_pre,
                            speech_lengths,
                            others,
                        )
                    else:
                        tf_ref, tf_pre = self._get_speech_masks(
                            criterion,
                            feature_mix,
                            noise_ref,
                            signal_ref,
                            signal_pre,
                            speech_lengths,
                            others,
                        )
                else:
                    # compute on spectrum
                    tf_ref = [self.encoder(sr, speech_lengths)[0] for sr in sref]
                    tf_pre = [self.encoder(sp, speech_lengths)[0] for sp in spre]

                l, s, o = loss_wrapper(tf_ref, tf_pre, {**others, **o})
            else:
                raise NotImplementedError("Unsupported loss type: %s" % str(criterion))

            loss += l * loss_wrapper.weight
            stats.update(s)

        if self.training and isinstance(loss, float):
            raise AttributeError(
                "At least one criterion must satisfy: only_for_test=False"
            )
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        batch_size = speech_ref[0].shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _align_ref_pre_channels(self, ref, pre, ch_dim=2, force_1ch=False):
        if ref is None or pre is None:
            return ref, pre
        # NOTE: input must be a list of time-domain signals
        index = ref[0].new_tensor(self.ref_channel, dtype=torch.long)

        # for models like SVoice that output multiple lists of separated signals
        pre_is_multi_list = isinstance(pre[0], (list, tuple))
        pre_dim = pre[0][0].dim() if pre_is_multi_list else pre[0].dim()

        if ref[0].dim() > pre_dim:
            # multi-channel reference and single-channel output
            ref = [r.index_select(ch_dim, index).squeeze(ch_dim) for r in ref]
        elif ref[0].dim() < pre_dim:
            # single-channel reference and multi-channel output
            if pre_is_multi_list:
                pre = [
                    p.index_select(ch_dim, index).squeeze(ch_dim)
                    for plist in pre
                    for p in plist
                ]
            else:
                pre = [p.index_select(ch_dim, index).squeeze(ch_dim) for p in pre]
        elif ref[0].dim() == pre_dim == 3 and force_1ch:
            # multi-channel reference and output
            ref = [r.index_select(ch_dim, index).squeeze(ch_dim) for r in ref]
            if pre_is_multi_list:
                pre = [
                    p.index_select(ch_dim, index).squeeze(ch_dim)
                    for plist in pre
                    for p in plist
                ]
            else:
                pre = [p.index_select(ch_dim, index).squeeze(ch_dim) for p in pre]
        return ref, pre

    def _get_noise_masks(
        self, criterion, feature_mix, speech_ref, noise_ref, noise_pre, ilens, others
    ):
        speech_spec = self.encoder(sum(speech_ref), ilens)[0]
        masks_ref = criterion.create_mask_label(
            feature_mix,
            [self.encoder(nr, ilens)[0] for nr in noise_ref],
            noise_spec=speech_spec,
        )
        if "mask_noise1" in others:
            masks_pre = [
                others["mask_noise{}".format(n + 1)] for n in range(self.num_noise_type)
            ]
        else:
            assert len(noise_pre) == len(noise_ref), (len(noise_pre), len(noise_ref))
            masks_pre = criterion.create_mask_label(
                feature_mix,
                [self.encoder(np, ilens)[0] for np in noise_pre],
                noise_spec=speech_spec,
            )
        return masks_ref, masks_pre

    def _get_dereverb_masks(
        self, criterion, feat_mix, noise_ref, dereverb_ref, dereverb_pre, ilens, others
    ):
        if noise_ref is not None:
            noise_spec = self.encoder(sum(noise_ref), ilens)[0]
        else:
            noise_spec = None
        masks_ref = criterion.create_mask_label(
            feat_mix,
            [self.encoder(dr, ilens)[0] for dr in dereverb_ref],
            noise_spec=noise_spec,
        )
        if "mask_dereverb1" in others:
            masks_pre = [
                others["mask_dereverb{}".format(spk + 1)]
                for spk in range(self.num_spk)
                if "mask_dereverb{}".format(spk + 1) in others
            ]
            assert len(masks_pre) == len(masks_ref), (len(masks_pre), len(masks_ref))
        else:
            assert len(dereverb_pre) == len(dereverb_ref), (
                len(dereverb_pre),
                len(dereverb_ref),
            )
            masks_pre = criterion.create_mask_label(
                feat_mix,
                [self.encoder(dp, ilens)[0] for dp in dereverb_pre],
                noise_spec=noise_spec,
            )
        return masks_ref, masks_pre

    def _get_speech_masks(
        self, criterion, feature_mix, noise_ref, speech_ref, speech_pre, ilens, others
    ):
        if noise_ref is not None:
            noise_spec = self.encoder(sum(noise_ref), ilens)[0]
        else:
            noise_spec = None
        masks_ref = criterion.create_mask_label(
            feature_mix,
            [self.encoder(sr, ilens)[0] for sr in speech_ref],
            noise_spec=noise_spec,
        )
        if "mask_spk1" in others:
            masks_pre = [
                others["mask_spk{}".format(spk + 1)] for spk in range(self.num_spk)
            ]
        else:
            masks_pre = criterion.create_mask_label(
                feature_mix,
                [self.encoder(sp, ilens)[0] for sp in speech_pre],
                noise_spec=noise_spec,
            )
        return masks_ref, masks_pre

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
