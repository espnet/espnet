"""Enhancement model module."""
from distutils.version import LooseVersion
from functools import reduce
import itertools
import logging
from multiprocessing.sharedctypes import Value
import random
from typing import Dict
from typing import List
from typing import OrderedDict
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.enh.loss.wrappers.mixit_solver import MixITSolver
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel as ESPnetEnhancementModelSupervised


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
        with_supervised_learning: bool = True,
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

        setattr(self, "loss_wrapper", None)
        for wrapper in loss_wrappers:
            # Supervised loss solver
            if isinstance(wrapper, FixedOrderSolver):
                self.supervised_loss_wrapper = wrapper
            elif isinstance(wrapper, PITSolver):
                self.supervised_loss_wrapper = wrapper
            # Unsupervised loss solver
            elif isinstance(wrapper, MixITSolver):
                self.unsupervised_loss_wrapper = wrapper
            else:
                raise ValueError("The model only supports"
                                 "the FixedOrderSolver and PITSolver for supervised loss solver, "
                                 "and MixIT Solver for unsupervised loss solver currently."
                                 )

        self.with_supervised_learning = with_supervised_learning

        assert getattr(self, "unsupervised_loss_wrapper", None) is not None, "Unsupervised loss is required."
        if self.with_supervised_learning:
            assert getattr(self, "supervised_loss_wrapper", None) is not None, "Supervised loss is required."

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
        speech_ref = []
        for spk in range(self.num_spk):
            if "speech_ref" + str(spk + 1) in kwargs:
                speech_ref.append(kwargs["speech_ref{}".format(spk + 1)])
            else:
                break
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)
        num_spk = speech_ref.shape[1]
        # Get utterance ids for generating mixture of mixtures in MixIT
        utt_ids = kwargs.get("utt_id", None)
        assert utt_ids is not None

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
                for n in range(num_spk)
                if "dereverb_ref{}".format(n + 1) in kwargs
            ]
            assert len(dereverb_speech_ref) in (1, num_spk), len(
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
        speech_ref = speech_ref[:, :, : speech_lengths.max()]
        speech_mix = speech_mix[:, : speech_lengths.max()]

        # TODO(xkc09): Hacking method
        if utt_ids[0].endswith("_supervised"):  # e.g. 911_6880-911-128684-0038_6880-216547-0054_unsupervised
            loss_wrapper_type = "pit"
            # loss_wrapper_type = "mixit"
        else :                 # e.g. 011_012-011c0213_0.85538_012o030a_-0.85538_supervised
            loss_wrapper_type = "mixit"

        if loss_wrapper_type == "pit":
            speech_ref = torch.cat(
                [
                    speech_ref,
                    torch.zeros_like(speech_ref),   # Set the rest of the reference to be all 0s.
                ],
                dim=1
            )
        elif loss_wrapper_type == "mixit":
            (
                speech_mix,
                speech_lengths,
                speech_ref,
            ) = self._generate_mixture_of_mixtures(
                speech_mix,
                speech_lengths,
                speech_ref,
                utt_ids,
            )
        else:
            raise NotImplementedError(f"{loss_wrapper_type} is not supported")

        # model forward
        speech_pre, feature_mix, feature_pre, others = self.forward_enhance(speech_mix, speech_lengths)

        # loss computation
        loss, stats, weight = self.forward_loss(
            speech_pre,
            speech_lengths,
            feature_mix,
            feature_pre,
            others,
            speech_ref.unbind(1),   # [(Batch, Length, ...)] x num_spk
            noise_ref,
            dereverb_speech_ref,
            loss_wrapper_type,
        )

        return loss, stats, weight

    def _generate_mixture_of_mixtures(
        self,
        speech_mix,
        speech_lengths,
        speech_ref,
        utt_ids,
    ):
        """Generate mixture of mixture for MixIT

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            utt_ids: List of strings, (Batch), utterance ids
            dereverb_speech_ref: (Batch, N, samples)
                        or (Batch, num_speaker, samples, channels)
            noise_ref: (Batch, num_noise_type, samples)
                        or (Batch, num_speaker, samples, channels)
            cal_loss: whether to calculate enh loss, default is True

        Returns: speech_mix, speech_ref_unsupervised, speech_ref_supervised
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref_unsupervised: (Batch, num_), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            speech_ref_supervised: (Batch_new, num_speaker, samples)
                        or (Batch_new, num_speaker, samples, channels)
        """
        def check_speaker_coexisting(spkr_lst1, spkr_lst2):
            same_spkrs = set(spkr_lst1) & set(spkr_lst2)
            return len(same_spkrs) == 0

        batch_size = speech_mix.shape[0]

        # TODO: add utt2speaker in the input, because Libri2Mix may not have such patterns, speaker id been separated by "_"
        # Get the speakers of each utterance
        spkrs = []
        for uttid in utt_ids:
            lst = uttid.split("_")
            spkrs.append((lst[0], lst[1]))

        mixture_of_batch_idx = []
        for idx in itertools.combinations(range(batch_size), 2):
            non_coexisting_speakers = check_speaker_coexisting(spkrs[idx[0]], spkrs[idx[1]])
            if non_coexisting_speakers:
                mixture_of_batch_idx.append(idx)

        speech_mom, speech_ref_unsupervised = [], []
        speech_mom_lengths, speech_lengths_unsupervised = [], []
        if len(mixture_of_batch_idx) > 0:
            if batch_size < len(mixture_of_batch_idx):
                mixture_of_batch_idx = random.sample(mixture_of_batch_idx, k=batch_size)
            for item in mixture_of_batch_idx:
                # MoMs
                speech_mom.append(speech_mix[item[0]] + speech_mix[item[1]])
                speech_mom_lengths.append(max(speech_lengths[item[0]], speech_lengths[item[1]]))

                # Unsupervised reference
                speech_ref_unsupervised.append(
                    torch.stack(
                        [speech_mix[item[0]], speech_mix[item[1]]],
                        axis=0,
                    )  # (num_mixture_of_mixtures, samples, ...)
                )
            speech_mom = torch.stack(speech_mom, axis=0)  # (batch, num_mixture_of_mixtures, samples, ...)
            speech_mom_lengths = torch.stack(speech_mom_lengths, axis=0)  # (batch, )
            speech_ref_unsupervised = torch.stack(speech_ref_unsupervised, axis=0)  # (batch, num_mixture_of_mixtures, samples, ...)
        else:
            speech_mom = speech_mix
            speech_mom_lengths = speech_lengths
            speech_ref_unsupervised = torch.stack([speech_mix, torch.zeros_like(speech_mix)], dim=1)
            logging.info(f"Couldn't generate MOM from these utterances {utt_ids}, just use two speaker mixture as input and reference")

        return speech_mom, speech_mom_lengths, speech_ref_unsupervised

    def forward_loss(
        self,
        speech_pre: torch.Tensor,
        speech_lengths: torch.Tensor,
        feature_mix: torch.Tensor,
        feature_pre: torch.Tensor,
        others: OrderedDict,
        speech_ref: List[torch.Tensor],
        noise_ref: torch.Tensor = None,
        dereverb_speech_ref: torch.Tensor = None,
        loss_wrapper_type: str = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        loss = 0.0
        stats = dict()
        o = {}

        for loss_wrapper in self.loss_wrappers:
            if loss_wrapper_type != loss_wrapper.type:
                continue

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
            if isinstance(criterion, TimeDomainLoss):
                loss_name = criterion.name.replace("_loss", "")
                stats[f"{loss_name}_{loss_wrapper.type}"] = -loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        batch_size = speech_ref[0].shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
