<<<<<<< HEAD
from functools import reduce
=======
from itertools import combinations
>>>>>>> refactor espnet2/enh/espnet_model.py to make it more flexible; update egs2/dirha_wsj/enh1
from itertools import permutations
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

<<<<<<< HEAD
=======

class ESPnetEnhancementModel(AbsESPnetModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self, enh_model: Optional[AbsEnhancement],
    ):
        assert check_argument_types()

        super().__init__()

        self.enh_model = enh_model
        self.num_spk = enh_model.num_spk
        self.num_noise_type = getattr(self.enh_model, "num_noise_type", 1)
        # get mask type for TF-domain models
        self.mask_type = getattr(self.enh_model, "mask_type", None)
        # get loss type for model training
        self.loss_type = getattr(self.enh_model, "loss_type", None)
        assert self.loss_type in (
            # mse_loss(predicted_mask, target_label)
            "mask_mse",
            # mse_loss(enhanced_magnitude_spectrum, target_magnitude_spectrum)
            "magnitude",
            # mse_loss(enhanced_complex_spectrum, target_complex_spectrum)
            "spectrum",
            # si_snr(enhanced_waveform, target_waveform)
            "si_snr",
        ), self.loss_type
        # for multi-channel signal
        self.ref_channel = getattr(self.enh_model, "ref_channel", -1)

    @staticmethod
    def _create_mask_label(mix_spec, ref_spec, mask_type="IAM"):
        """Create mask label.

        :param mix_spec: ComplexTensor(B, T, F)
        :param ref_spec: [ComplexTensor(B, T, F), ...] or ComplexTensor(B, T, F)
        :param noise_spec: ComplexTensor(B, T, F)
        :return: [Tensor(B, T, F), ...] or [ComplexTensor(B, T, F), ...]
        """

        assert mask_type in [
            "IBM",
            "IRM",
            "IAM",
            "PSM",
            "NPSM",
            "PSM^2",
        ], f"mask type {mask_type} not supported"
        eps = 10e-8
        mask_label = []
        for r in ref_spec:
            mask = None
            if mask_type == "IBM":
                flags = [abs(r) >= abs(n) for n in ref_spec]
                mask = reduce(lambda x, y: x * y, flags)
                mask = mask.int()
            elif mask_type == "IRM":
                # TODO(Wangyou): need to fix this,
                #  as noise referecens are provided separately
                mask = abs(r) / (sum(([abs(n) for n in ref_spec])) + eps)
            elif mask_type == "IAM":
                mask = abs(r) / (abs(mix_spec) + eps)
                mask = mask.clamp(min=0, max=1)
            elif mask_type == "PSM" or mask_type == "NPSM":
                phase_r = r / (abs(r) + eps)
                phase_mix = mix_spec / (abs(mix_spec) + eps)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                    phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r) / (abs(mix_spec) + eps)) * cos_theta
                mask = (
                    mask.clamp(min=0, max=1)
                    if mask_type == "NPSM"
                    else mask.clamp(min=-1, max=1)
                )
            elif mask_type == "PSM^2":
                # This is for training beamforming masks
                phase_r = r / (abs(r) + eps)
                phase_mix = mix_spec / (abs(mix_spec) + eps)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                    phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r).pow(2) / (abs(mix_spec).pow(2) + eps)) * cos_theta
                mask = mask.clamp(min=-1, max=1)
            assert mask is not None, f"mask type {mask_type} not supported"
            mask_label.append(mask)
        return mask_label

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

        # dereverberated noisy signal
        # (optional, only used for frontend models with WPE)
        dereverb_speech_ref = kwargs.get("dereverb_ref", None)

        batch_size = speech_mix.shape[0]
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int() * speech_mix.shape[1]
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

        loss, speech_pre, mask_pre, out_lengths, perm = self._compute_loss(
            speech_mix,
            speech_lengths,
            speech_ref,
            dereverb_speech_ref=dereverb_speech_ref,
            noise_ref=noise_ref,
        )

        # add stats for logging
        if self.loss_type != "si_snr":
            if self.training:
                si_snr = None
            else:
                speech_pre = [
                    self.enh_model.stft.inverse(ps, speech_lengths)[0]
                    for ps in speech_pre
                ]
                if speech_ref[0].dim() == 3:
                    # For si_snr loss, only select one channel as the reference
                    speech_ref = [sr[..., self.ref_channel] for sr in speech_ref]
                # compute si-snr loss
                si_snr_loss, perm = self._permutation_loss(
                    speech_ref, speech_pre, self.si_snr_loss, perm=perm
                )
                si_snr = -si_snr_loss.detach()

            stats = dict(
                si_snr=si_snr,
                loss=loss.detach(),
            )
        else:
            stats = dict(si_snr=-loss.detach(), loss=loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _compute_loss(
        self,
        speech_mix,
        speech_lengths,
        speech_ref,
        dereverb_speech_ref=None,
        noise_ref=None,
        cal_loss=True,  # weather to calculate enh loss
    ):
        """Compute loss according to self.loss_type.

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            dereverb_speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            noise_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)

        Returns:
            loss: (torch.Tensor) speech enhancement loss
            speech_pre: (List[torch.Tensor] or List[ComplexTensor]) enhanced speech or spectrum(s)
            mask_pre: (OrderedDict) estimated masks or None
            output_lengths: (Batch,)
            perm: () best permutation
        """
        if self.loss_type != "si_snr":
            # prepare reference speech and reference spectrum
            speech_ref = torch.unbind(speech_ref, dim=1)
            spectrum_ref = [self.enh_model.stft(sr)[0] for sr in speech_ref]

            # List[ComplexTensor(Batch, T, F)] or List[ComplexTensor(Batch, T, C, F)]
            spectrum_ref = [
                ComplexTensor(sr[..., 0], sr[..., 1]) for sr in spectrum_ref
            ]
            spectrum_mix = self.enh_model.stft(speech_mix)[0]
            spectrum_mix = ComplexTensor(spectrum_mix[..., 0], spectrum_mix[..., 1])

            # predict separated speech and masks
            spectrum_pre, tf_length, mask_pre = self.enh_model(
                speech_mix, speech_lengths
            )

            if not cal_loss:
                loss, perm = None, None
                return loss, spectrum_pre, mask_pre, tf_length, perm

            # compute TF masking loss
            if self.loss_type == "magnitude":
                # compute loss on magnitude spectrum
                magnitude_pre = [abs(ps) for ps in spectrum_pre]
                if spectrum_ref[0].dim() > magnitude_pre[0].dim():
                    # only select one channel as the reference
                    magnitude_ref = [
                        abs(sr[..., self.ref_channel, :]) for sr in spectrum_ref
                    ]
                else:
                    magnitude_ref = [abs(sr) for sr in spectrum_ref]

                tf_loss, perm = self._permutation_loss(
                    magnitude_ref, magnitude_pre, self.tf_mse_loss
                )
            elif self.loss_type == "spectrum":
                # compute loss on complex spectrum
                if spectrum_ref[0].dim() > spectrum_pre[0].dim():
                    # only select one channel as the reference
                    spectrum_ref = [sr[..., self.ref_channel, :] for sr in spectrum_ref]

                tf_loss, perm = self._permutation_loss(
                    spectrum_ref, spectrum_pre, self.tf_mse_loss
                )
            elif self.loss_type.startswith("mask"):
                if self.loss_type == "mask_mse":
                    loss_func = self.tf_mse_loss
                else:
                    raise ValueError("Unsupported loss type: %s" % self.loss_type)

                assert mask_pre is not None
                mask_pre_ = [
                    mask_pre["spk{}".format(spk + 1)] for spk in range(self.num_spk)
                ]

                # prepare ideal masks
                mask_ref = self._create_mask_label(
                    spectrum_mix, spectrum_ref, mask_type=self.mask_type
                )

                # compute TF masking loss
                tf_loss, perm = self._permutation_loss(mask_ref, mask_pre_, loss_func)

                if "dereverb" in mask_pre:
                    if dereverb_speech_ref is None:
                        raise ValueError(
                            "No dereverberated reference for training!\n"
                            'Please specify "--use_dereverb_ref true" in run.sh'
                        )

                    dereverb_spectrum_ref = self.enh_model.stft(dereverb_speech_ref)[0]
                    dereverb_spectrum_ref = ComplexTensor(
                        dereverb_spectrum_ref[..., 0], dereverb_spectrum_ref[..., 1]
                    )
                    # ComplexTensor(B, T, F) or ComplexTensor(B, T, C, F)
                    dereverb_mask_ref = self._create_mask_label(
                        spectrum_mix, [dereverb_spectrum_ref], mask_type=self.mask_type
                    )[0]

                    tf_loss = (
                        tf_loss
                        + loss_func(dereverb_mask_ref, mask_pre["dereverb"]).mean()
                    )

                if "noise1" in mask_pre:
                    if noise_ref is None:
                        raise ValueError(
                            "No noise reference for training!\n"
                            'Please specify "--use_noise_ref true" in run.sh'
                        )

                    noise_ref = torch.unbind(noise_ref, dim=1)
                    noise_spectrum_ref = [
                        self.enh_model.stft(nr)[0] for nr in noise_ref
                    ]
                    noise_spectrum_ref = [
                        ComplexTensor(nr[..., 0], nr[..., 1])
                        for nr in noise_spectrum_ref
                    ]
                    noise_mask_ref = self._create_mask_label(
                        spectrum_mix, noise_spectrum_ref, mask_type=self.mask_type
                    )

                    mask_noise_pre = [
                        mask_pre["noise{}".format(n + 1)]
                        for n in range(self.num_noise_type)
                    ]
                    tf_noise_loss, perm_n = self._permutation_loss(
                        noise_mask_ref, mask_noise_pre, loss_func
                    )
                    tf_loss = tf_loss + tf_noise_loss
            else:
                raise ValueError("Unsupported loss type: %s" % self.loss_type)

            loss = tf_loss
            return loss, spectrum_pre, mask_pre, tf_length, perm

        else:
            if speech_ref.dim() == 4:
                # For si_snr loss of multi-channel input,
                # only select one channel as the reference
                speech_ref = speech_ref[..., self.ref_channel]

            speech_pre, speech_lengths, *__ = self.enh_model.forward_rawwav(
                speech_mix, speech_lengths
            )
            if not cal_loss:
                loss, perm = None, None
                return loss, speech_pre, None, speech_lengths, perm

            # speech_pre: list[(batch, sample)]
            assert speech_pre[0].dim() == 2, speech_pre[0].dim()
            speech_ref = torch.unbind(speech_ref, dim=1)

            # compute si-snr loss
            si_snr_loss, perm = self._permutation_loss(
                speech_ref, speech_pre, self.si_snr_loss_zeromean
            )
            loss = si_snr_loss

            return loss, speech_pre, None, speech_lengths, perm

    @staticmethod
    def tf_mse_loss(ref, inf):
        """time-frequency MSE loss.

        :param ref: (Batch, T, F) or (Batch, T, C, F)
        :param inf: (Batch, T, F) or (Batch, T, C, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if ref.dim() == 3:
            mseloss = (abs(ref - inf) ** 2).mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = (abs(ref - inf) ** 2).mean(dim=[1, 2, 3])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )

        return mseloss

    @staticmethod
    def tf_l1_loss(ref, inf):
        """time-frequency L1 loss.

        :param ref: (Batch, T, F) or (Batch, T, C, F)
        :param inf: (Batch, T, F) or (Batch, T, C, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if ref.dim() == 3:
            l1loss = abs(ref - inf).mean(dim=[1, 2])
        elif ref.dim() == 4:
            l1loss = abs(ref - inf).mean(dim=[1, 2, 3])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        return l1loss

    @staticmethod
    def si_snr_loss(ref, inf):
        """si-snr loss

        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        ref = ref / torch.norm(ref, p=2, dim=1, keepdim=True)
        inf = inf / torch.norm(inf, p=2, dim=1, keepdim=True)

        s_target = (ref * inf).sum(dim=1, keepdims=True) * ref
        e_noise = inf - s_target

        si_snr = 20 * torch.log10(
            torch.norm(s_target, p=2, dim=1) / torch.norm(e_noise, p=2, dim=1)
        )
        return -si_snr

    @staticmethod
    def si_snr_loss_zeromean(ref, inf):
        """si_snr loss with zero-mean in pre-processing.

        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        eps = 1e-8

        assert ref.size() == inf.size()
        B, T = ref.size()
        # mask padding position along T

        # Step 1. Zero-mean norm
        mean_target = torch.sum(ref, dim=1, keepdim=True) / T
        mean_estimate = torch.sum(inf, dim=1, keepdim=True) / T
        zero_mean_target = ref - mean_target
        zero_mean_estimate = inf - mean_estimate

        # Step 2. SI-SNR with order
        # reshape to use broadcast
        s_target = zero_mean_target  # [B, T]
        s_estimate = zero_mean_estimate  # [B, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=1, keepdim=True)  # [B, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=1, keepdim=True) + eps  # [B, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, T]

        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + eps
        )
        # print('pair_si_snr',pair_wise_si_snr[0,:])
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + eps)  # [B]
        # print(pair_wise_si_snr)

        return -1 * pair_wise_si_snr

    @staticmethod
    def _permutation_loss(ref, inf, criterion, perm=None):
        """The basic permutation loss function.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm (torch.Tensor): specified permutation (batch, num_spk)
        Returns:
            loss (torch.Tensor): (batch)
            perm (torch.Tensor): (batch, num_spk)
                                 e.g. tensor([[1, 0, 2], [0, 1, 2]])
        """
        assert len(ref) == len(inf), (len(ref), len(inf))
        num_spk = len(ref)

        def pair_loss(permutation):
            return sum(
                [criterion(ref[s], inf[t]) for s, t in enumerate(permutation)]
            ) / len(permutation)

        if perm is None:
            device = ref[0].device
            all_permutations = list(permutations(range(num_spk)))
            losses = torch.stack([pair_loss(p) for p in all_permutations], dim=1)
            loss, perm = torch.min(losses, dim=1)
            perm = torch.index_select(
                torch.tensor(all_permutations, device=device, dtype=torch.long),
                0,
                perm,
            )
        else:
            loss = torch.tensor(
                [
                    torch.tensor(
                        [
                            criterion(
                                ref[s][batch].unsqueeze(0), inf[t][batch].unsqueeze(0)
                            )
                            for s, t in enumerate(p)
                        ]
                    ).mean()
                    for batch, p in enumerate(perm)
                ]
            )

        return loss.mean(), perm

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
>>>>>>> refactor espnet2/enh/espnet_model.py to make it more flexible; update egs2/dirha_wsj/enh1


class ESPnetEnhancementModel_mixIT(ESPnetEnhancementModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self,
        enh_model: Optional[AbsEnhancement],
    ):
        assert check_argument_types()

        super().__init__()

        self.enh_model = enh_model
        self.num_spk = enh_model.num_spk
        self.num_noise_type = getattr(self.enh_model, "num_noise_type", 1)
        # get mask type for TF-domain models
        self.mask_type = getattr(self.enh_model, "mask_type", None)
        # for multi-channel signal
        self.ref_channel = getattr(self.enh_model, "ref_channel", -1)

        # For mixIT setting
        # self.sup_ratio= enh_model.sup_ratio
        # self.unsup_1spk_ratio= enh_model.unsup_1spk_ratio
        # self.unsup_2spk_ratio= enh_model.unsup_2spk_ratio
        self.sup_ratio = 0.0
        self.unsup_1spk_ratio = 0.5
        self.unsup_2spk_ratio = 0.5
        self.bs_limit = 4
        assert self.sup_ratio + self.unsup_1spk_ratio + self.unsup_2spk_ratio == 1

    def forward(
        self,
        speech_mix: torch.Tensor,
        uttids: list,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            uttids: list : (Batch)
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
        """
        # clean speech signal of each speaker
        speech_ref = [kwargs["speech_ref{}".format(spk + 1)] for spk in range(2)]
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

        # dereverberated noisy signal
        # (optional, only used for frontend models with WPE)
        dereverb_speech_ref = kwargs.get("dereverb_ref", None)

        batch_size = speech_mix.shape[0]
        mix_uttids = uttids
        assert len(mix_uttids) == batch_size
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int() * speech_mix.shape[1]
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )
        batch_size = speech_mix.shape[0]

        # for data-parallel
        speech_ref = speech_ref[:, :, : speech_lengths.max()]
        speech_mix = speech_mix[:, : speech_lengths.max()]

        sup_mixtures, unsup_1spk_mixtures, unsup_2spk_mixtures = self.uttids_split(
            uttids, speech_mix, speech_ref
        )

        if (
            len(unsup_1spk_mixtures) + len(unsup_2spk_mixtures) > 1
        ):  # there is 2 or more unsup mixtures
            mix_of_mixtures, mix_ref = self.get_mix_of_mixtures_spk1or2(
                unsup_1spk_mixtures, unsup_2spk_mixtures, self.bs_limit
            )  # (Batch',T) (Batch',2,T)
            if mix_of_mixtures is not None:
                # print(mix_of_mixtures.shape,mix_ref.shape)
                mix_speech_lengths = (
                    torch.ones(mix_of_mixtures.shape[0]).int() * speech_mix.shape[1]
                )
            else:
                print("No mix of mixtures generated.")
        else:
            mix_of_mixtures = None

        if not (
            isinstance(self.enh_model, TasNet) or isinstance(self.enh_model, DPRNN)
        ):
            # prepare reference speech and reference spectrum
            speech_ref = torch.unbind(speech_ref, dim=1)
            spectrum_ref = [self.enh_model.stft(sr)[0] for sr in speech_ref]

            # List[ComplexTensor(Batch, T, F)] or List[ComplexTensor(Batch, T, C, F)]
            spectrum_ref = [
                ComplexTensor(sr[..., 0], sr[..., 1]) for sr in spectrum_ref
            ]
            spectrum_mix = self.enh_model.stft(speech_mix)[0]
            spectrum_mix = ComplexTensor(spectrum_mix[..., 0], spectrum_mix[..., 1])

            # prepare ideal masks
            mask_ref = self._create_mask_label(
                spectrum_mix, spectrum_ref, mask_type=self.mask_type
            )

            if dereverb_speech_ref is not None:
                dereverb_spectrum_ref = self.enh_model.stft(dereverb_speech_ref)[0]
                dereverb_spectrum_ref = ComplexTensor(
                    dereverb_spectrum_ref[..., 0], dereverb_spectrum_ref[..., 1]
                )
                # ComplexTensor(B, T, F) or ComplexTensor(B, T, C, F)
                dereverb_mask_ref = self._create_mask_label(
                    spectrum_mix, [dereverb_spectrum_ref], mask_type=self.mask_type
                )[0]

            if noise_ref is not None:
                noise_ref = torch.unbind(noise_ref, dim=1)
                noise_spectrum_ref = [self.enh_model.stft(nr)[0] for nr in noise_ref]
                noise_spectrum_ref = [
                    ComplexTensor(nr[..., 0], nr[..., 1]) for nr in noise_spectrum_ref
                ]
                noise_mask_ref = self._create_mask_label(
                    spectrum_mix, noise_spectrum_ref, mask_type=self.mask_type
                )

            # predict separated speech and masks
            spectrum_pre, tf_length, mask_pre = self.enh_model(
                speech_mix, speech_lengths
            )

            # TODO:Chenda, Shall we add options for computing loss on
            #  the masked spectrum?
            # compute TF masking loss
            if mask_pre is None:
                # compute loss on magnitude spectrum instead
                magnitude_pre = [abs(ps) for ps in spectrum_pre]
                magnitude_ref = [abs(sr) for sr in spectrum_ref]
                tf_loss, perm = self._permutation_loss(
                    magnitude_ref, magnitude_pre, self.tf_mse_loss
                )
            else:
                mask_pre_ = [
                    mask_pre["spk{}".format(spk + 1)] for spk in range(self.num_spk)
                ]

                # compute TF masking loss
                # TODO: Chenda, Shall we add options for
                #  computing loss on the masked spectrum?
                tf_loss, perm = self._permutation_loss(
                    mask_ref, mask_pre_, self.tf_mse_loss
                )

                if "dereverb" in mask_pre:
                    if dereverb_speech_ref is None:
                        raise ValueError(
                            "No dereverberated reference for training!\n"
                            'Please specify "--use_dereverb_ref true" in run.sh'
                        )
                    tf_loss = (
                        tf_loss
                        + self.tf_l1_loss(
                            dereverb_mask_ref, mask_pre["dereverb"]
                        ).mean()
                    )

                if "noise1" in mask_pre:
                    if noise_ref is None:
                        raise ValueError(
                            "No noise reference for training!\n"
                            'Please specify "--use_noise_ref true" in run.sh'
                        )
                    mask_noise_pre = [
                        mask_pre["noise{}".format(n + 1)]
                        for n in range(self.num_noise_type)
                    ]
                    tf_noise_loss, perm_n = self._permutation_loss(
                        noise_mask_ref, mask_noise_pre, self.tf_mse_loss
                    )
                    tf_loss = tf_loss + tf_noise_loss

            if self.training:
                si_snr = None
            else:
                speech_pre = [
                    self.enh_model.stft.inverse(ps, speech_lengths)[0]
                    for ps in spectrum_pre
                ]
                if speech_ref[0].dim() == 3:
                    # For si_snr loss, only select one channel as the reference
                    speech_ref = [sr[..., self.ref_channel] for sr in speech_ref]
                # compute si-snr loss
                si_snr_loss, perm = self._permutation_loss(
                    speech_ref, speech_pre, self.si_snr_loss, perm=perm
                )
                si_snr = -si_snr_loss.detach()

            loss = tf_loss

            stats = dict(
                si_snr=si_snr,
                loss=loss.detach(),
            )
        else:
            if speech_ref.dim() == 4:
                # For si_snr loss of multi-channel input,
                # only select one channel as the reference
                speech_ref = speech_ref[..., self.ref_channel]

            if sup_mixtures:
                # print('sup mix:', sup_mixtures)
                speech_mix = torch.stack([d[1] for d in sup_mixtures], dim=0)
                speech_ref = torch.stack([d[2] for d in sup_mixtures], dim=0)  # BS,2,T
                # Add 2 more all-zero references to train the PIT loss.
                speech_ref = torch.cat(
                    [speech_ref, 0 * speech_ref], dim=1
                )  # BS,4,T for PIT loss

                speech_pre, *__ = self.enh_model.forward_rawwav(speech_mix, None)

                # speech_pre: list[(batch, sample)]
                assert speech_pre[0].dim() == 2, speech_pre[0].dim()
                speech_ref = torch.unbind(speech_ref, dim=1)

                # compute si-snr loss
                si_snr_loss, perm = self._permutation_loss(
                    speech_ref, speech_pre, self.si_snr_loss_zeromean
                )
                si_snr = -si_snr_loss
                sup_PIT_loss = si_snr_loss
            else:
                sup_PIT_loss = 0

            if mix_of_mixtures is not None:
                speech_pre_MoM, speech_lengths, *__ = self.enh_model.forward_rawwav(
                    mix_of_mixtures, mix_speech_lengths
                )
                mix_ref = mix_ref.transpose(0, 1)  # 2,Batch',T
                # speech_pre_MoM: list[(batch, sample)] length of M
                speech_pre_MoM = torch.stack(speech_pre_MoM, dim=0)  # M,Batch',T
                # print('mix_ref,speech_pre:',mix_ref.shape,speech_pre_MoM.shape)
                si_snr_loss_MoM, perm_MoM = self._mixIT_loss(
                    mix_ref, speech_pre_MoM, self.mixIT_si_snr_loss_zeromean
                )
                # import soundfile as sf
                # sf.write('mom.wav',mix_of_mixtures[0].data.cpu().numpy(),8000)
                # sf.write('pre0.wav',speech_pre_MoM[:,0].data.cpu().numpy().T,8000)
                # print('si_MOM,per_Mom',si_snr_loss_MoM,perm_MoM)
                unsup_mixIT_loss = si_snr_loss_MoM
            else:
                unsup_mixIT_loss = 0
            loss = sup_PIT_loss + unsup_mixIT_loss
            print("pit,mixIT:", sup_PIT_loss, unsup_mixIT_loss)
            stats = dict(
                loss=loss.detach(),
                pit_loss=sup_PIT_loss.detach() if sup_PIT_loss != 0 else 0,
                mixIT_loss=unsup_mixIT_loss.detach() if unsup_mixIT_loss != 0 else 0,
            )
            print(stats)
            # stats = dict(loss=loss.detach(),)
            # force_gatherable: to-device and to-tensor if scalar for DataParallel
            loss, stats, weight = force_gatherable(
                (loss, stats, batch_size), speech_mix.device
            )
            return loss, stats, weight

    def uttids_split(self, uttids, speech_mix, speech_ref):
        """Split all the mixtures from WSJ0_2mix with uttids

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            uttids: list : (Batch)
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
        """
        # convert to percent range
        self.sup_ratio *= 100
        self.unsup_1spk_ratio *= 100
        self.unsup_2spk_ratio *= 100

        speech_mix = speech_mix.unbind(dim=0)  # list[ samples ]
        speech_ref = speech_ref.unbind(dim=0)  # list[ number_spk,samples]
        sup_ratio, unsup_1spk, unsup_2spk = [], [], []
        for uttid, mix, ref in zip(uttids, speech_mix, speech_ref):
            # uttid example: '407_024_407a010z_1.0657_024o030v_-1.0657'
            uttid_split_number = abs(float(uttid.split("_")[-1]) * 1e04) % 100  # [0~99]
            if uttid_split_number < self.sup_ratio:
                # print('sup utt:',uttid)
                sup_ratio.append([uttid, mix, ref])
            elif (
                self.sup_ratio
                <= uttid_split_number
                < (self.sup_ratio + self.unsup_1spk_ratio)
            ):
                unsup_1spk.append([uttid, mix, ref])
                # print('1spk utt:',uttid)
            elif (self.sup_ratio + self.unsup_1spk_ratio) <= uttid_split_number < 100:
                unsup_2spk.append([uttid, mix, ref])
                # print('2spk utt:', uttid)
        return sup_ratio, unsup_1spk, unsup_2spk

    @staticmethod
    def get_mix_of_mixtures_spk1or2(unsup_1spk, unsup_2spk, bs_limit=True):
        """
        :param unsup_1spk: List[uttids,mix,ref]
        :param unsup_2spk: List[uttids,mix,ref]
        :return: mix_of_mixtures(Batch', T)
                 mix_ref(Batch', 2, T)
        """
        # Choose one channel as the unsup 1spk mixture
        if len(unsup_1spk):
            uttids_1spk = [d[0] for d in unsup_1spk]
            speech_mix_1spk = torch.stack(
                [d[1] for d in unsup_1spk], dim=0
            )  # BS_1spk,T
            speech_ref_1spk = torch.stack(
                [d[2] for d in unsup_1spk], dim=0
            )  # BS_1spk,2,T
            uttids_1spk_new = []
            speech_mix_1spk_new = []
            speech_ref_1spk_new = []
            for id, mix, ref in zip(uttids_1spk, speech_mix_1spk, speech_ref_1spk):
                spks = id.split("_")[:2]
                chosen_idx = random.randint(0, 1)  # 0 or 1
                chosen_spk = spks[chosen_idx]
                uttids_1spk_new.append(
                    "{}_{}".format(chosen_spk, chosen_spk)
                )  # like '010_010'
                speech_mix_1spk_new.append(ref[chosen_idx])  # list[T]
                ref[1 - chosen_idx] = 0  # set another channel to zero
                speech_ref_1spk_new.append(ref)  # list[2,T]
            uttids_1spk = uttids_1spk_new
            speech_mix_1spk = torch.stack(
                [d for d in speech_mix_1spk_new], dim=0
            )  # BS_1spk,T
            speech_ref_1spk = torch.stack(
                [d for d in speech_ref_1spk_new], dim=0
            )  # BS_1spk,2,T
        else:
            uttids_1spk = []
            speech_mix_1spk = []
            speech_ref_1spk = []

        # import soundfile as sf
        # sf.write('1spk_mix.wav',speech_mix_1spk_new[0].data.cpu().numpy(),8000)
        # sf.write('1spk_ref.wav',speech_ref_1spk_new[0].data.cpu().numpy().T,8000)

        if len(unsup_2spk):
            # pack 2spk mixture
            uttids_2spk = [d[0] for d in unsup_2spk]
            speech_mix_2spk = torch.stack(
                [d[1] for d in unsup_2spk], dim=0
            )  # BS_2spk,T
            speech_ref_2spk = torch.stack(
                [d[2] for d in unsup_2spk], dim=0
            )  # BS_2spk,2,T
        else:
            uttids_2spk = []
            speech_mix_2spk = []
            speech_ref_2spk = []

        batch_size = len(unsup_1spk) + len(unsup_2spk)
        uttids = uttids_1spk + uttids_2spk
        if (type(speech_mix_1spk) is not list) and (type(speech_mix_2spk) is not list):
            # print('speech_mix:',speech_mix_1spk,speech_mix_2spk)
            speech_mix = torch.cat([speech_mix_1spk, speech_mix_2spk], dim=0)  # BS,T
            speech_ref = torch.cat([speech_ref_1spk, speech_ref_2spk], dim=0)  # BS,2,T
        if (type(speech_mix_1spk) is not list) and (type(speech_mix_2spk) is list):
            speech_mix = speech_mix_1spk  # BS,T
        if (type(speech_mix_1spk) is list) and (type(speech_mix_2spk) is not list):
            speech_mix = speech_mix_2spk  # BS,T

        spks_list = [uttid.split("_")[:2] for uttid in uttids]
        mix_of_mixtures_list = []
        mix_ref1, mix_ref2 = [], []
        for p in combinations(range(batch_size), 2):
            spk1_list = spks_list[p[0]]
            spk2_list = spks_list[p[1]]
            same_spk_list = [x for x in spk1_list if x in spk2_list]  # get same spk
            if not same_spk_list:  # if no same spk
                # print('mix of mixtures', spk1_list, spk2_list)
                mix_of_mixtures_tmp = speech_mix[p[0]] + speech_mix[p[1]]
                mix_of_mixtures_list.append(mix_of_mixtures_tmp)
                mix_ref1.append(speech_mix[p[0]])
                mix_ref2.append(speech_mix[p[1]])
        if not mix_of_mixtures_list:  # no mix-of-mixtures generated sucessfully here
            return None, None
        mix_ref1 = torch.stack(mix_ref1, dim=0)  # Batch', T
        mix_ref2 = torch.stack(mix_ref2, dim=0)  # Batch', T
        mix_of_mixtures = torch.stack(mix_of_mixtures_list, dim=0)
        mix_ref = torch.stack([mix_ref1, mix_ref2], dim=1)
        if bs_limit:
            # TODO(Jing): should be uniform sampling
            # print('mix_ref:',mix_ref.shape)
            return mix_of_mixtures[:bs_limit], mix_ref[:bs_limit]
        else:
            return mix_of_mixtures, mix_ref

    @staticmethod
    def get_mix_of_mixtures(uttids, speech_mix, speech_ref, bs_limit=True):
        """
        :param uttids: (Batch)
        :param speech_mix: (Batch, T)
        :param speech_ref: (Batch, num_spk, T)
        :return: mix_of_mixtures(Batch', T)
                 mix_ref(Batch', 2, T)
        """
        batch_size = len(uttids)
        spks_list = [uttid.split("_")[:2] for uttid in uttids]
        mix_of_mixtures_list = []
        mix_ref1, mix_ref2 = [], []
        from itertools import combinations

        for p in combinations(range(batch_size), 2):
            spk1_list = spks_list[p[0]]
            spk2_list = spks_list[p[1]]
            same_spk_list = [x for x in spk1_list if x in spk2_list]  # get same spk
            if not same_spk_list:  # if no same spk
                # print('mix of mixtures', spk1_list, spk2_list)
                mix_of_mixtures_tmp = speech_mix[p[0]] + speech_mix[p[1]]
                mix_of_mixtures_list.append(mix_of_mixtures_tmp)
                mix_ref1.append(speech_mix[p[0]])
                mix_ref2.append(speech_mix[p[1]])
        if not mix_of_mixtures_list:  # no mix-of-mixtures generated sucessfully here
            return None, None
        mix_ref1 = torch.stack(mix_ref1, dim=0)  # Batch', T
        mix_ref2 = torch.stack(mix_ref2, dim=0)  # Batch', T
        mix_of_mixtures = torch.stack(mix_of_mixtures_list, dim=0)
        mix_ref = torch.stack([mix_ref1, mix_ref2], dim=1)
        if bs_limit and len(mix_of_mixtures_list) > batch_size:
            # TODO(Jing): should be uniform sampling
            # print('mix_ref:',mix_ref.shape)
            return mix_of_mixtures[:batch_size], mix_ref[:batch_size]
        else:
            return mix_of_mixtures, mix_ref

    @staticmethod
    def mixIT_si_snr_loss_zeromean(ref, inf, SNR_max=30):
        """
        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        eps = 1e-8
        alpha = 10 ** (-1 * float(SNR_max) / 10)

        assert ref.size() == inf.size()
        B, T = ref.size()
        # mask padding position along T

        # Step 1. Zero-mean norm
        mean_target = torch.sum(ref, dim=1, keepdim=True) / T
        mean_estimate = torch.sum(inf, dim=1, keepdim=True) / T
        zero_mean_target = ref - mean_target
        zero_mean_estimate = inf - mean_estimate

        # Step 2. SI-SNR with order
        # reshape to use broadcast
        s_target = zero_mean_target  # [B, T]
        s_estimate = zero_mean_estimate  # [B, T]
        # s_target = <s', s>s / ||s||^2
        s_target_energy = torch.sum(s_target ** 2, dim=1, keepdim=True)  # [B, 1]
        # e_noise = s' - s_target
        e_noise = s_target - s_estimate  # [B, T]
        e_noise_energy = torch.sum(e_noise ** 2, dim=1)  # [B, T]

        # SI-SNR = 10 * log_10(||s_target||^2 / (||e_noise||^2 + alpha*||s_target||^2)
        pair_wise_si_snr = s_target_energy / (
            e_noise_energy + alpha * s_target_energy + eps
        )
        # print('pair_si_snr',pair_wise_si_snr[0,:])
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + eps)  # [B]
        # print('pair wise:',pair_wise_si_snr.shape)
        return -1 * pair_wise_si_snr

    @staticmethod
    def _mixIT_loss(ref, inf, criterion=mixIT_si_snr_loss_zeromean, perm=None):
        """
        Args:
            ref (torch.Tensor): [2, batch', T, ...]
            inf (torch.Tensor): [M, batch', T, ...]
            criterion (function): Loss function
            perm: (batch)
        Returns:
            torch.Tensor: (batch)
        """
        num_aim = len(ref)
        assert num_aim == 2
        M = len(inf)
        idx_list = [[0, 1] for __ in range(M)]

        def pair_loss(permutation):
            """
            :param permutation: tuple like (0,0,1,...,1) with length of M
            :return:
            """
            first_row = torch.FloatTensor(permutation).to(ref.device)
            mixing_matrix = (
                torch.stack([first_row, 1 - first_row], dim=0)
                .unsqueeze(0)
                .expand(inf.shape[1], 2, M)
            )  # Binary matrix of [batch',2,M]
            mixing_infs = torch.bmm(
                mixing_matrix, inf.transpose(0, 1)
            )  # [1,2,M]*[batch',M,T] -->[batch',2,T]
            # print('mix_matrix,mixing_infs',mixing_matrix.shape,mixing_infs.shape)
            return sum([criterion(ref[t], mixing_infs[:, t]) for t in range(2)]) / 2

        losses = torch.stack([pair_loss(p) for p in product(*idx_list)], dim=1)
        if perm is None:
            loss, perm = torch.min(losses, dim=1)
            # print(losses.shape)
            # print(losses)
        else:
            loss = losses[torch.arange(losses.shape[0]), perm]

        return loss.mean(), perm
<<<<<<< HEAD

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
=======
>>>>>>> refactor espnet2/enh/espnet_model.py to make it more flexible; update egs2/dirha_wsj/enh1
