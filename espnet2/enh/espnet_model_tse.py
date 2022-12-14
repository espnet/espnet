"""Enhancement model module."""
from typing import Dict, List, OrderedDict, Tuple

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

EPS = torch.finfo(torch.get_default_dtype()).eps


class ESPnetExtractionModel(AbsESPnetModel):
    """Target Speaker Extraction Frontend model"""

    def __init__(
        self,
        encoder: AbsEncoder,
        extractor: AbsExtractor,
        decoder: AbsDecoder,
        loss_wrappers: List[AbsLossWrapper],
        num_spk: int = 1,
        share_encoder: bool = True,
    ):
        assert check_argument_types()

        super().__init__()

        self.encoder = encoder
        self.extractor = extractor
        self.decoder = decoder
        # Whether to share encoder for both mixyture and enrollment
        self.share_encoder = share_encoder
        self.num_spk = num_spk

        self.loss_wrappers = loss_wrappers
        names = [w.criterion.name for w in self.loss_wrappers]
        if len(set(names)) != len(names):
            raise ValueError("Duplicated loss names are not allowed: {}".format(names))
        for w in self.loss_wrappers:
            if getattr(w.criterion, "is_noise_loss", False):
                raise ValueError("is_noise_loss=True is not supported")
            elif getattr(w.criterion, "is_dereverb_loss", False):
                raise ValueError("is_dereverb_loss=True is not supported")

        # for multi-channel signal
        self.ref_channel = getattr(self.extractor, "ref_channel", -1)

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
            enroll_ref2: (Batch, samples_aux)
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

        assert "enroll_ref1" in kwargs, "At least 1 enrollment signal is required."
        # enrollment signal for each speaker (as the target)
        enroll_ref = [
            # (Batch, samples_aux)
            kwargs["enroll_ref{}".format(spk + 1)]
            for spk in range(self.num_spk)
            if "enroll_ref{}".format(spk + 1) in kwargs
        ]
        enroll_ref_lengths = [
            # (Batch,)
            kwargs.get(
                "enroll_ref{}_lengths".format(spk + 1),
                torch.ones(batch_size).int().fill_(enroll_ref[spk].size(1)),
            )
            for spk in range(self.num_spk)
            if "enroll_ref{}".format(spk + 1) in kwargs
        ]

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
        for aux in enroll_ref:
            assert aux.shape[0] == speech_mix.shape[0], (aux.shape, speech_mix.shape)

        # for data-parallel
        speech_ref = speech_ref[..., : speech_lengths.max()].unbind(dim=1)

        speech_mix = speech_mix[:, : speech_lengths.max()]
        enroll_ref = [
            enroll_ref[spk][:, : enroll_ref_lengths[spk].max()]
            for spk in range(len(enroll_ref))
        ]

        # model forward
        speech_pre, feature_mix, feature_pre, others = self.forward_enhance(
            speech_mix, speech_lengths, enroll_ref, enroll_ref_lengths
        )

        # loss computation
        loss, stats, weight, perm = self.forward_loss(
            speech_pre,
            speech_lengths,
            feature_mix,
            feature_pre,
            others,
            speech_ref,
        )
        return loss, stats, weight

    def forward_enhance(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        enroll_ref: torch.Tensor,
        enroll_ref_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_mix, flens = self.encoder(speech_mix, speech_lengths)
        if self.share_encoder:
            feature_aux, flens_aux = zip(
                *[
                    self.encoder(enroll_ref[spk], enroll_ref_lengths[spk])
                    for spk in range(len(enroll_ref))
                ]
            )
        else:
            feature_aux = enroll_ref
            flens_aux = enroll_ref_lengths

        feature_pre, _, others = zip(
            *[
                self.extractor(
                    feature_mix,
                    flens,
                    feature_aux[spk],
                    flens_aux[spk],
                    suffix_tag=f"_spk{spk + 1}",
                )
                for spk in range(len(enroll_ref))
            ]
        )
        others = {k: v for dic in others for k, v in dic.items()}
        if feature_pre[0] is not None:
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        loss = 0.0
        stats = {}
        o = {}
        perm = None
        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            if getattr(criterion, "only_for_test", False) and self.training:
                continue

            if isinstance(criterion, TimeDomainLoss):
                assert speech_pre is not None
                sref, spre = self._align_ref_pre_channels(
                    speech_ref, speech_pre, ch_dim=2, force_1ch=True
                )
                # for the time domain criterions
                l, s, o = loss_wrapper(sref, spre, {**others, **o})
            elif isinstance(criterion, FrequencyDomainLoss):
                sref, spre = self._align_ref_pre_channels(
                    speech_ref, speech_pre, ch_dim=2, force_1ch=False
                )
                # for the time-frequency domain criterions
                if criterion.compute_on_mask:
                    # compute loss on masks
                    tf_ref, tf_pre = self._get_speech_masks(
                        criterion,
                        feature_mix,
                        None,
                        speech_ref,
                        speech_pre,
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

            if perm is None and "perm" in o:
                perm = o["perm"]

        if self.training and isinstance(loss, float):
            raise AttributeError(
                "At least one criterion must satisfy: only_for_test=False"
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
