"""Enhancement model module."""
import contextlib
from typing import Dict, List, Optional, OrderedDict, Tuple

import numpy as np
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
from espnet2.enh.separator.uses_separator import USESSeparator
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
        flexible_numspk: bool = False,
        extract_feats_in_collect_stats: bool = False,
        normalize_variance: bool = False,
        normalize_variance_per_ch: bool = False,
        categories: list = [],
        category_weights: list = [],
    ):
        """Main entry of speech enhancement/separation model training.

        Args:
            encoder: waveform encoder that converts waveforms to feature representations
            separator: separator that enhance or separate the feature representations
            decoder: waveform decoder that converts the feature back to waveforms
            mask_module: mask module that converts the feature to masks
                NOTE: Only used for compatibility with joint speaker diarization.
                See test/espnet2/enh/test_espnet_enh_s2t_model.py for details.
            loss_wrappers: list of loss wrappers
                Each loss wrapper contains a criterion for loss calculation and
                the corresonding loss weight.
                The losses will be calculated in the order of the list and summed up.
            ------------------------------------------------------------------
            stft_consistency: (deprecated, kept for compatibility) whether to compute
                the TF-domain loss while enforcing STFT consistency
                NOTE: STFT consistency is now always used for frequency-domain spectrum
                losses.
            loss_type: (deprecated, kept for compatibility) loss type
            mask_type: (deprecated, kept for compatibility) mask type in TF-domain model
            ------------------------------------------------------------------
            flexible_numspk: whether to allow the model to predict a variable number of
                speakers in its output.
                NOTE: This should be used when training a speech separation model for
                unknown number of speakers.
            ------------------------------------------------------------------
            extract_feats_in_collect_stats: used in espnet2/tasks/abs_task.py for
                determining whether or not to skip model building in collect_stats stage
                (stage 5 in egs2/*/enh1/enh.sh).
            normalize_variance: whether to normalize the signal variance before model
                forward, and revert it back after.
            normalize_variance_per_ch: whether to normalize the signal variance for each
                channel instead of the whole signal.
                NOTE: normalize_variance and normalize_variance_per_ch cannot be True
                at the same time.
            ------------------------------------------------------------------
            categories: list of all possible categories of minibatches (order matters!)
                (e.g. ["1ch_8k_reverb", "1ch_8k_both"] for multi-condition training)
                NOTE: this will be used to convert category index to the corresponding
                name for logging in forward_loss.
                Different categories will have different loss name suffixes.
            category_weights: list of weights for each category.
                Used to set loss weights for batches of different categories.
        """
        assert check_argument_types()

        super().__init__()

        self.encoder = encoder
        self.separator = separator
        self.decoder = decoder
        self.mask_module = mask_module
        self.num_spk = separator.num_spk
        # If True, self.num_spk is regarded as the MAXIMUM possible number of speakers
        self.flexible_numspk = flexible_numspk
        self.num_noise_type = getattr(self.separator, "num_noise_type", 1)

        self.loss_wrappers = loss_wrappers
        names = [w.criterion.name for w in self.loss_wrappers]
        if len(set(names)) != len(names):
            raise ValueError("Duplicated loss names are not allowed: {}".format(names))

        # kept for compatibility
        self.mask_type = mask_type.upper() if mask_type else None
        self.loss_type = loss_type
        self.stft_consistency = stft_consistency

        # for multi-channel signal
        self.ref_channel = getattr(self.separator, "ref_channel", None)
        if self.ref_channel is None:
            self.ref_channel = 0

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.normalize_variance = normalize_variance
        self.normalize_variance_per_ch = normalize_variance_per_ch
        if normalize_variance and normalize_variance_per_ch:
            raise ValueError(
                "normalize_variance and normalize_variance_per_ch cannot be True "
                "at the same time."
            )

        # list all possible categories of the batch (order matters!)
        # (used to convert category index to the corresponding name for logging)
        self.categories = {}
        if categories:
            count = 0
            for c in categories:
                if c not in self.categories:
                    self.categories[count] = c
                    count += 1
        # used to set loss weights for batches of different categories
        if category_weights:
            assert len(category_weights) == len(self.categories)
            self.category_weights = tuple(category_weights)
        else:
            self.category_weights = tuple(1.0 for _ in self.categories)

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
        # reference speech signal of each speaker
        assert "speech_ref1" in kwargs, "At least 1 reference signal input is required."
        speech_ref = [
            kwargs.get(
                f"speech_ref{spk + 1}",
                torch.zeros_like(kwargs["speech_ref1"]),
            )
            for spk in range(self.num_spk)
            if f"speech_ref{spk + 1}" in kwargs
        ]
        num_spk = len(speech_ref) if self.flexible_numspk else self.num_spk
        assert len(speech_ref) == num_spk, (len(speech_ref), num_spk)
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
                for n in range(num_spk)
                if "dereverb_ref{}".format(n + 1) in kwargs
            ]
            assert len(dereverb_speech_ref) in (1, num_spk), len(dereverb_speech_ref)
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

        # sampling frequency information about the batch
        fs = None
        if "utt2fs" in kwargs:
            # All samples must have the same sampling rate
            fs = kwargs["utt2fs"][0].item()
            assert all([fs == kwargs["utt2fs"][0].item() for fs in kwargs["utt2fs"]])

            # Adaptively adjust the STFT/iSTFT window/hop sizes for USESSeparator
            if not isinstance(self.separator, USESSeparator):
                fs = None

        # category information (integer) about the batch
        category = kwargs.get("utt2category", None)
        if (
            self.categories
            and category is not None
            and category[0].item() not in self.categories
        ):
            raise ValueError(f"Category '{category}' is not listed in self.categories")

        additional = {}
        # Additional data is required in Deep Attractor Network
        if isinstance(self.separator, DANSeparator):
            additional["feature_ref"] = [
                self.encoder(r, speech_lengths, fs=fs)[0] for r in speech_ref
            ]
        if self.flexible_numspk:
            additional["num_spk"] = num_spk
        # Additional information is required in USES for multi-condition training
        if category is not None and isinstance(self.separator, USESSeparator):
            cat = self.categories[category[0].item()]
            if cat.endswith("_both"):
                additional["mode"] = "both"
            elif cat.endswith("_reverb"):
                additional["mode"] = "dereverb"
            else:
                additional["mode"] = "no_dereverb"

        speech_mix = speech_mix[:, : speech_lengths.max()]

        ###################################
        # Normalize the signal variance
        if self.normalize_variance_per_ch:
            dim = 1
            mix_std_ = torch.std(speech_mix, dim=dim, keepdim=True)
            speech_mix = speech_mix / mix_std_  # RMS normalization
        elif self.normalize_variance:
            if speech_mix.ndim > 2:
                dim = (1, 2)
            else:
                dim = 1
            mix_std_ = torch.std(speech_mix, dim=dim, keepdim=True)
            speech_mix = speech_mix / mix_std_  # RMS normalization

        # model forward
        speech_pre, feature_mix, feature_pre, others = self.forward_enhance(
            speech_mix, speech_lengths, additional, fs=fs
        )

        ###################################
        # De-normalize the signal variance
        if self.normalize_variance_per_ch and speech_pre is not None:
            if mix_std_.ndim > 2:
                mix_std_ = mix_std_[:, :, self.ref_channel]
            speech_pre = [sp * mix_std_ for sp in speech_pre]
        elif self.normalize_variance and speech_pre is not None:
            if mix_std_.ndim > 2:
                mix_std_ = mix_std_.squeeze(2)
            speech_pre = [sp * mix_std_ for sp in speech_pre]

        # loss computation
        loss, stats, weight, perm = self.forward_loss(
            speech_pre,
            speech_lengths,
            feature_mix,
            feature_pre,
            others,
            speech_ref,
            noise_ref,
            dereverb_speech_ref,
            category,
            num_spk=num_spk,
            fs=fs,
        )
        return loss, stats, weight

    def forward_enhance(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        additional: Optional[Dict] = None,
        fs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_mix, flens = self.encoder(speech_mix, speech_lengths, fs=fs)
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
            # for models like SVoice that output multiple lists of separated signals
            pre_is_multi_list = isinstance(feature_pre[0], (list, tuple))
            if pre_is_multi_list:
                speech_pre = [
                    [self.decoder(p, speech_lengths, fs=fs)[0] for p in ps]
                    for ps in feature_pre
                ]
            else:
                speech_pre = [
                    self.decoder(ps, speech_lengths, fs=fs)[0] for ps in feature_pre
                ]
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
        feature_pre: List[torch.Tensor],
        others: OrderedDict,
        speech_ref: List[torch.Tensor],
        noise_ref: Optional[List[torch.Tensor]] = None,
        dereverb_speech_ref: Optional[List[torch.Tensor]] = None,
        category: Optional[torch.Tensor] = None,
        num_spk: Optional[int] = None,
        fs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        # for calculating loss on estimated noise signals
        if getattr(self.separator, "predict_noise", False):
            assert "noise1" in others, others.keys()
        if noise_ref is not None and "noise1" in others:
            for n in range(self.num_noise_type):
                key = "noise{}".format(n + 1)
                others[key] = self.decoder(others[key], speech_lengths, fs=fs)[0]
        # for calculating loss on dereverberated signals
        if getattr(self.separator, "predict_dereverb", False):
            assert "dereverb1" in others, others.keys()
        if dereverb_speech_ref is not None and "dereverb1" in others:
            for spk in range(num_spk if num_spk else self.num_spk):
                key = "dereverb{}".format(spk + 1)
                if key in others:
                    others[key] = self.decoder(others[key], speech_lengths, fs=fs)[0]

        loss = speech_ref[0].new_tensor(0.0)
        stats = {}
        o = {}
        perm = None
        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            only_for_test = getattr(criterion, "only_for_test", False)
            if only_for_test and self.training:
                continue
            is_noise_loss = getattr(criterion, "is_noise_loss", False)
            is_dereverb_loss = getattr(criterion, "is_dereverb_loss", False)
            if is_noise_loss:
                if noise_ref is None:
                    raise ValueError(
                        "No noise reference for training!\n"
                        'Please specify "--use_noise_ref true" in run.sh'
                    )
                signal_ref = noise_ref
                signal_pre = [
                    others["noise{}".format(n + 1)] for n in range(self.num_noise_type)
                ]
            elif is_dereverb_loss:
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

            zero_weight = loss_wrapper.weight == 0.0
            if isinstance(criterion, TimeDomainLoss):
                if signal_pre is None:
                    if is_noise_loss or is_dereverb_loss:
                        # Skip loss computation for noise/dereverb-specific losses
                        # if no noise/dereverb signals are predicted for this batch
                        if category is not None:
                            for idx, c in self.categories.items():
                                stats[criterion.name + "_" + c] = torch.full_like(
                                    loss, np.nan
                                )
                        else:
                            stats[criterion.name] = torch.full_like(loss, np.nan)
                        continue
                    raise ValueError(
                        "Predicted waveform is required for time-domain loss."
                    )
                sref, spre = self._align_ref_pre_channels(
                    signal_ref, signal_pre, ch_dim=2, force_1ch=True
                )
                # for the time domain criterions
                with torch.no_grad() if zero_weight else contextlib.ExitStack():
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
                            fs=fs,
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
                            fs=fs,
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
                            fs=fs,
                        )
                else:
                    # compute on spectrum
                    tf_ref = [self.encoder(sr, speech_lengths, fs=fs)[0] for sr in sref]
                    # for models like SVoice that output multiple lists of
                    # separated signals
                    pre_is_multi_list = isinstance(spre[0], (list, tuple))
                    if pre_is_multi_list:
                        tf_pre = [
                            [self.encoder(sp, speech_lengths, fs=fs)[0] for sp in ps]
                            for ps in spre
                        ]
                    else:
                        tf_pre = [
                            self.encoder(sp, speech_lengths, fs=fs)[0] for sp in spre
                        ]

                with torch.no_grad() if zero_weight else contextlib.ExitStack():
                    l, s, o = loss_wrapper(tf_ref, tf_pre, {**others, **o})
            else:
                raise NotImplementedError("Unsupported loss type: %s" % str(criterion))

            loss += l * loss_wrapper.weight

            # rename the loss keys with a category prefix
            if (
                self.categories
                and category is not None
                and category[0].item() not in self.categories
            ):
                raise ValueError(
                    f"Category '{category}' is not listed in self.categories"
                )
            if category is not None:
                for idx, c in self.categories.items():
                    if idx == category[0].item():
                        s[criterion.name + "_" + c] = s.pop(criterion.name)
                    else:
                        s[criterion.name + "_" + c] = torch.full_like(loss, np.nan)
            else:
                idx = 0
            stats.update(s)
            loss *= self.category_weights[idx] if self.category_weights else 1.0

            if perm is None and "perm" in o:
                perm = o["perm"]

        if self.training and not loss.requires_grad:
            raise AttributeError(
                "Loss must be a tensor with gradient in the training mode. "
                "Please check the following:\n"
                "1. At least one criterion must satisfy: only_for_test=False"
                "2. At least one criterion must always be computed in the training mode"
                " regardless of is_noise_loss=True or is_dereverb_loss=True"
            )
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        batch_size = speech_ref[0].shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight, perm

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
        self,
        criterion,
        feature_mix,
        speech_ref,
        noise_ref,
        noise_pre,
        ilens,
        others,
        fs=None,
    ):
        speech_spec = self.encoder(sum(speech_ref), ilens, fs=fs)[0]
        masks_ref = criterion.create_mask_label(
            feature_mix,
            [self.encoder(nr, ilens, fs=fs)[0] for nr in noise_ref],
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
                [self.encoder(np, ilens, fs=fs)[0] for np in noise_pre],
                noise_spec=speech_spec,
            )
        return masks_ref, masks_pre

    def _get_dereverb_masks(
        self,
        criterion,
        feat_mix,
        noise_ref,
        dereverb_ref,
        dereverb_pre,
        ilens,
        others,
        fs=None,
    ):
        if noise_ref is not None:
            noise_spec = self.encoder(sum(noise_ref), ilens, fs=fs)[0]
        else:
            noise_spec = None
        masks_ref = criterion.create_mask_label(
            feat_mix,
            [self.encoder(dr, ilens, fs=fs)[0] for dr in dereverb_ref],
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
                [self.encoder(dp, ilens, fs=fs)[0] for dp in dereverb_pre],
                noise_spec=noise_spec,
            )
        return masks_ref, masks_pre

    def _get_speech_masks(
        self,
        criterion,
        feature_mix,
        noise_ref,
        speech_ref,
        speech_pre,
        ilens,
        others,
        fs=None,
    ):
        if noise_ref is not None:
            noise_spec = self.encoder(sum(noise_ref), ilens, fs=fs)[0]
        else:
            noise_spec = None
        masks_ref = criterion.create_mask_label(
            feature_mix,
            [self.encoder(sr, ilens, fs=fs)[0] for sr in speech_ref],
            noise_spec=noise_spec,
        )
        if "mask_spk1" in others:
            masks_pre = [
                others["mask_spk{}".format(spk + 1)]
                for spk in range(self.num_spk)
                if "mask_spk{}".format(spk + 1) in others
            ]
        else:
            masks_pre = criterion.create_mask_label(
                feature_mix,
                [self.encoder(sp, ilens, fs=fs)[0] for sp in speech_pre],
                noise_spec=noise_spec,
            )
        return masks_ref, masks_pre

    @staticmethod
    def sort_by_perm(nn_output, perm):
        """Sort the input list of tensors by the specified permutation.

        Args:
            nn_output: List[torch.Tensor(Batch, ...)], len(nn_output) == num_spk
            perm: (Batch, num_spk) or List[torch.Tensor(num_spk)]
        Returns:
            nn_output_new: List[torch.Tensor(Batch, ...)]
        """
        if len(nn_output) == 1:
            return nn_output
        # (Batch, num_spk, ...)
        nn_output = torch.stack(nn_output, dim=1)
        if not isinstance(perm, torch.Tensor):
            # perm is a list or tuple
            perm = torch.stack(perm, dim=0)
        assert nn_output.size(1) == perm.size(1), (nn_output.shape, perm.shape)
        diff_dim = nn_output.dim() - perm.dim()
        if diff_dim > 0:
            perm = perm.view(*perm.shape, *[1 for _ in range(diff_dim)]).expand_as(
                nn_output
            )
        return torch.gather(nn_output, 1, perm).unbind(dim=1)

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
