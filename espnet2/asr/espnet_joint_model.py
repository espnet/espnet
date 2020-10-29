from functools import reduce
from itertools import permutations
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetEnhASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        enh: Optional[AbsEnhancement],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        ctc: CTC,
        rnnt_decoder: None,
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        enh_weight: float = 0.5,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= enh_weight <= 1.0, ctc_weight
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.enh_weight = enh_weight
        self.token_list = token_list.copy()

        self.enh_model = enh
        self.num_spk = enh.num_spk
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

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.decoder = decoder
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        self.rnnt_decoder = rnnt_decoder
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

        # TODO(Jing): find out the -1 or 0 here
        # self.idx_blank = token_list.index(sym_blank) # 0
        self.idx_blank = -1

    def _create_mask_label(self, mix_spec, ref_spec, mask_type="IAM"):
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
                    if mask_label == "NPSM"
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
        speech_mix_lengths: torch.Tensor,
        speech_ref1: torch.Tensor,
        speech_ref2: torch.Tensor,
        text_ref1: torch.Tensor,
        text_ref2: torch.Tensor,
        text_ref1_lengths: torch.Tensor,
        text_ref2_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Enhancement + Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_ref1_lengths.dim() == text_ref2_lengths.dim() == 1, (
            text_ref1_lengths.shape,
            text_ref2_lengths.shape,
        )
        # Check that batch_size is unified
        assert (
            speech_mix.shape[0]
            == speech_mix_lengths.shape[0]
            == text_ref1.shape[0]
            == text_ref1_lengths.shape[0]
            == text_ref2.shape[0]
            == text_ref2_lengths.shape[0]
        ), (
            speech_mix.shape,
            speech_mix_lengths.shape,
            text_ref1.shape,
            text_ref1_lengths.shape,
        )
        batch_size = speech_mix.shape[0]

        # for data-parallel
        text_length_max = max(text_ref1_lengths.max(), text_ref2_lengths.max())
        text_ref1 = torch.cat(
            [
                text_ref1,
                torch.ones(batch_size, text_length_max, dtype=text_ref1.dtype).to(
                    text_ref1.device
                )
                * self.idx_blank,
            ],
            dim=1,
        )
        text_ref2 = torch.cat(
            [
                text_ref2,
                torch.ones(batch_size, text_length_max, dtype=text_ref1.dtype).to(
                    text_ref1.device
                )
                * self.idx_blank,
            ],
            dim=1,
        )
        text_ref1 = text_ref1[:, :text_length_max]
        text_ref2 = text_ref2[:, :text_length_max]

        # 0. Enhancement
        # make sure the speech_pre is the raw waveform with same size.
        loss_enh, perm, speech_pre = self.forward_enh(
            speech_mix,
            speech_mix_lengths,
            speech_ref1=speech_ref1,
            speech_ref2=speech_ref2,
        )
        # speech_pre: (bs,num_spk,T)
        assert speech_pre[:, 0].shape == speech_mix.shape

        # Pack the separated speakers into the ASR part.
        speech_pre_all = speech_pre.view(-1, speech_mix.shape[-1])  # (bs*num_spk, T)
        speech_pre_lengths = torch.stack(
            [speech_mix_lengths, speech_mix_lengths], dim=1
        ).view(-1)
        text_ref_all = torch.stack([text_ref1, text_ref2], dim=1).view(
            batch_size * 2, -1
        )
        text_ref_lengths = torch.stack(
            [text_ref1_lengths, text_ref2_lengths], dim=1
        ).view(-1)

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech_pre_all, speech_pre_lengths)

        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text_ref_all, text_ref_lengths
            )

        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text_ref_all, text_ref_lengths
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(
                encoder_out, encoder_out_lens, text_ref_all, text_ref_lengths
            )

        if self.ctc_weight == 0.0:
            loss_asr = loss_att
        elif self.ctc_weight == 1.0:
            loss_asr = loss_ctc
        else:
            loss_asr = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        if self.enh_weight == 0.0:
            loss_enh = None
            loss = loss_asr
        else:
            loss = (1 - self.enh_weight) * loss_asr + self.enh_weight * loss_enh

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            loss_enh=loss_enh.detach() if loss_enh is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]
        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        # 1. Extract feats
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)

        # 2. Data augmentation for spectrogram
        if self.specaug is not None and self.training:
            feats, feats_lengths = self.specaug(feats, feats_lengths)

        # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError

    # Enhancement related, basicly from the espnet2/enh/espnet_model.py
    def forward_enh(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        resort_pre: bool = True,
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

            # TODO(Chenda), Shall we add options for computing loss on
            #  the masked spectrum?
            # compute TF masking loss
            if self.loss_type == "magnitude":
                # compute loss on magnitude spectrum
                magnitude_pre = [abs(ps) for ps in spectrum_pre]
                magnitude_ref = [abs(sr) for sr in spectrum_ref]
                tf_loss, perm = self._permutation_loss(
                    magnitude_ref, magnitude_pre, self.tf_mse_loss
                )
            elif self.loss_type == "spectrum":
                # compute loss on complex spectrum
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

            if spectrum_pre is None and self.loss_type == "mask":
                # Need the wav prediction in training
                # TODO(Jing): should coordinate with the enh/nets/***, this is ugly now.
                self.enh_model.training = False
                speech_pre, *__ = self.enh_model.forward_rawwav(
                    speech_mix, speech_lengths
                )
                self.enh_model.training = self.training
            else:
                speech_pre, *__ = self.enh_model.forward_rawwav(
                    speech_mix, speech_lengths
                )

            loss = tf_loss

        else:
            if speech_ref.dim() == 4:
                # For si_snr loss of multi-channel input,
                # only select one channel as the reference
                speech_ref = speech_ref[..., self.ref_channel]

            speech_pre, speech_lengths, *__ = self.enh_model.forward_rawwav(
                speech_mix, speech_lengths
            )
            # speech_pre: list[(batch, sample)]
            assert speech_pre[0].dim() == 2, speech_pre[0].dim()
            speech_ref = torch.unbind(speech_ref, dim=1)

            # compute si-snr loss
            si_snr_loss, perm = self._permutation_loss(
                speech_ref, speech_pre, self.si_snr_loss_zeromean
            )
            loss = si_snr_loss

        if resort_pre:
            # speech_pre : list[(bs,T)] of spk
            # perm : list[(num_spk)] of batch
            speech_pre_list = []
            for batch_idx, p in enumerate(perm):
                batch_list = []
                for spk_idx in p:
                    batch_list.append(speech_pre[spk_idx][batch_idx])  # spk,T
                speech_pre_list.append(torch.stack(batch_list, dim=0))

            speech_pre = torch.stack(speech_pre_list, dim=0)  # bs,num_spk,T
        else:
            speech_pre = torch.stack(speech_pre, dim=1)  # bs,num_spk,T

        return loss, perm, speech_pre

    @staticmethod
    def tf_mse_loss(ref, inf):
        """time-frequency MSE loss.

        :param ref: (Batch, T, F)
        :param inf: (Batch, T, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if ref.dim() == 3:
            mseloss = (abs(ref - inf) ** 2).mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = (abs(ref - inf) ** 2).mean(dim=[1, 2, 3])
        else:
            raise ValueError("Invalid input shape: ref={}, inf={}".format(ref, inf))

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
            raise ValueError("Invalid input shape: ref={}, inf={}".format(ref, inf))
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
            ref (List[torch.Tensor]): [(batch, ...), ...]
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm: (batch)
        Returns:
            loss: torch.Tensor: (batch)
            perm: list[(num_spk)]
        """
        num_spk = len(ref)

        def pair_loss(permutation):
            return sum(
                [criterion(ref[s], inf[t]) for s, t in enumerate(permutation)]
            ) / len(permutation)

        losses = torch.stack(
            [pair_loss(p) for p in permutations(range(num_spk))], dim=1
        )
        if perm is None:
            loss, perm = torch.min(losses, dim=1)
        else:
            loss = losses[torch.arange(losses.shape[0]), perm]

        perm_list = [p for p in permutations(range(num_spk))]
        perm_detail = []
        for p in perm:
            perm_detail.append(perm_list[p])  # egs: list([1,0]) or list([0,2,1])
        return loss.mean(), perm_detail
