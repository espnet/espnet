from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from scipy.optimize import linear_sum_assignment
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

<<<<<<< HEAD
=======

def fliter_attrs(a, b):
    a_attr = [attr for attr in a if attr[:2] != "__" and attr[-2:] != "__"]
    b_attr = [attr for attr in b if attr[:2] != "__" and attr[-2:] != "__"]
    a_attr_ = [i for i in a_attr if i not in b_attr]
    b_attr_ = [i for i in b_attr if i not in a_attr]
    return a_attr_, b_attr_

>>>>>>> update multi-speaker ASR task

class ESPnetEnhASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        enh_model: Optional[ESPnetEnhancementModel],
        asr_model: Optional[ESPnetASRModel],
        enh_weight: float = 0.5,
        enh_return_type: Union[str, None] = "waveform",
        cal_enh_loss: bool = True,
        end2end_train: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= asr_model.ctc_weight <= 1.0, asr_model.ctc_weight
        assert 0.0 <= enh_weight <= 1.0, asr_model.ctc_weight
        # TODO(Jing): add humanfriendly or str_or_none typeguard
        if enh_return_type == "none" or enh_return_type == "None":
            enh_return_type = None
        assert enh_return_type in ["waveform", "spectrum", None]
        assert asr_model.rnnt_decoder is None, "Not implemented"
        super().__init__()

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.enh_weight = enh_weight
        self.enh_return_type = enh_return_type  # 'waveform' or 'spectrum' or None
        self.cal_enh_loss = cal_enh_loss

        # TODO(Jing): find out the -1 or 0 here
        # self.idx_blank = token_list.index(sym_blank) # 0
        self.idx_blank = -1
        self.num_spk = enh_model.num_spk
        if self.num_spk > 1:
            assert (
                asr_model.ctc_weight != 0.0 or cal_enh_loss == True
            )  # need at least one to cal PIT permutation
        if enh_return_type == "waveform" or enh_return_type == None:
            assert (
                self.asr_subclass.frontend.apply_stft
            ), "need apply stft in asr fronend part"
        elif enh_return_type == "spectrum":
            # TODO(Xuankai,Jing): verify this additional uttMVN
            self.asr_subclass.additional_utt_mvn = UtteranceMVN(norm_means=True, norm_vars=False)
            assert (
                not self.asr_subclass.frontend.apply_stft
            ), "avoid usage of stft in asr fronend part"

    def _create_mask_label(self, mix_spec, ref_spec, mask_type="IAM"):
        """Create mask label.

        :param mix_spec: ComplexTensor(B, T, F)
        :param ref_spec: [ComplexTensor(B, T, F), ...] or ComplexTensor(B, T, F)
        :param noise_spec: ComplexTensor(B, T, F)
        :return: [Tensor(B, T, F), ...] or [ComplexTensor(B, T, F), ...]
        """

<<<<<<< HEAD
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
=======
        # fliter the specific attr for each subclass
        self.enh_attr, self.asr_attr = fliter_attrs(self.enh_attr, self.asr_attr)
        for arr in self.enh_attr:
            exec("self.{} = self.enh_subclass.{}".format(arr, arr))
        for arr in self.asr_attr:
            exec("self.{} = self.asr_subclass.{}".format(arr, arr))
>>>>>>> update multi-speaker ASR task

        self.enh_model.return_spec_in_training = True

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        text_ref1: torch.Tensor,
        text_ref2: torch.Tensor,
        text_ref1_lengths: torch.Tensor,
        text_ref2_lengths: torch.Tensor,
        speech_ref1: torch.Tensor = None,
        speech_ref2: torch.Tensor = None,
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
        if self.enh_return_type != None:
            # make sure the speech_pre is the raw waveform with same size.
            loss_enh, perm, speech_pre, speech_pre_lengths = self.forward_enh(
                speech_mix,
                speech_mix_lengths,
                speech_ref1=speech_ref1,
                speech_ref2=speech_ref2,
            )
            if self.enh_return_type == "waveform":
                # speech_pre: (bs,num_spk,T)
                assert speech_pre[:, 0].shape == speech_mix.shape

                if not self.end2end_train:
                    # if the End and ASR is trained independetly
                    # use the speech_ref to train asr
                    speech_pre = torch.stack([speech_ref1, speech_ref2], dim=1)

                # Pack the separated speakers into the ASR part.
                # TODO(Jing): unified order bs*spk or spk*bs
                speech_pre_all = speech_pre.view(
                    -1, speech_mix.shape[-1]
                )  # (bs*num_spk, T)
                speech_pre_lengths = torch.stack(
                    [speech_mix_lengths, speech_mix_lengths], dim=1
                ).view(-1)
                text_ref_all = torch.stack([text_ref1, text_ref2], dim=1).view(
                    batch_size * 2, -1
                )
                text_ref_lengths = torch.stack(
                    [text_ref1_lengths, text_ref2_lengths], dim=1
                ).view(-1)
            elif self.enh_return_type == "spectrum":
                # The return value speech_pre is actually the
                # spectrum List[torch.Tensor(B, T, D, 2)] or List[torch.complex(B, T, D)]
                if speech_pre[0].dtype == torch.tensor:
                    assert speech_pre[0].dim >= 4 and speech_pre[0].size(-1) == 2
                elif isinstance(speech_pre[0], ComplexTensor):
                    speech_pre = [
                        torch.stack([pre.real, pre.imag], dim=-1) for pre in speech_pre
                    ]
                    assert speech_pre[0].dim() >= 4 and speech_pre[0].size(-1) == 2
                speech_pre_all = torch.cat(speech_pre, dim=0)  # (N_spk*B, T, D)
                speech_pre_lengths = torch.cat([speech_pre_lengths, speech_pre_lengths])
                text_ref_all = torch.cat([text_ref1, text_ref2], dim=0)
                text_ref_lengths = torch.cat([text_ref1_lengths, text_ref2_lengths])
            else:
                raise NotImplementedError("No such enh_return_type")

            n_speaker_asr = 1  # single-channel asr after enh
        else:
            # Dont do enhancement
            speech_pre_all = speech_mix  # bs,T
            speech_pre_lengths = speech_mix_lengths  # bs
            text_ref_all = torch.stack([text_ref1, text_ref2], dim=1).view(
                batch_size * 2, -1
            )
            text_ref_lengths = torch.stack(
                [text_ref1_lengths, text_ref2_lengths], dim=1
            ).view(-1)
            n_speaker_asr = self.num_spk  # multi-channel asr w/o enh
            perm = None

            # TODO(Jing): Need the multi outputs setting from the Encoder
            raise NotImplementedError

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech_pre_all, speech_pre_lengths)

        # 2a. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            if (
                self.cal_enh_loss and perm != None
            ):  # Permutation was done in enhancement
                assert n_speaker_asr == 1
                loss_ctc, cer_ctc, _, _, = self._calc_ctc_loss_with_spk(
                    encoder_out,
                    encoder_out_lens,
                    text_ref_all,
                    text_ref_lengths,
                    n_speakers=1,
                )
            else:  # Permutation is determined by CTC
                (
                    loss_ctc,
                    cer_ctc,
                    encoder_out,
                    encoder_out_lens,
                ) = self._calc_ctc_loss_with_spk(
                    encoder_out,
                    encoder_out_lens,
                    text_ref_all,
                    text_ref_lengths,
                    n_speakers=self.num_spk,
                )

        # 2b. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
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

        if self.enh_weight == 0.0 or self.cal_enh_loss == False:
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

<<<<<<< HEAD
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

=======
>>>>>>> update multi-speaker ASR task
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
            kwargs["speech_ref{}".format(spk + 1)]
            if f"speech_ref{spk + 1}" in kwargs
            else None
            for spk in range(self.num_spk)
        ]
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        if speech_ref[0] is not None:
            speech_ref = torch.stack(speech_ref, dim=1)
        else:
            speech_ref = None

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

        if speech_ref == None and noise_ref == None and dereverb_speech_ref == None:
            # There is no ref provided, avoid the enh loss
            assert self.cal_enh_loss == False, (
                "There is no reference,"
                "cal_enh_loss must be false, but {} given.".format(self.cal_enh_loss)
            )

        batch_size = speech_mix.shape[0]
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int() * speech_mix.shape[1]
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        if speech_ref is not None:
            assert (
                speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0]
            ), (
                speech_mix.shape,
                speech_ref.shape,
                speech_lengths.shape,
            )

        # for data-parallel
        speech_ref = (
            speech_ref[:, :, : speech_lengths.max()] if speech_ref is not None else None
        )
        speech_mix = speech_mix[:, : speech_lengths.max()]

        loss, speech_pre, mask_pre, out_lengths, perm = self._compute_loss(
            speech_mix,
            speech_lengths,
            speech_ref,
            dereverb_speech_ref=dereverb_speech_ref,
            noise_ref=noise_ref,
            cal_loss=self.cal_enh_loss,  # weather to cal enh_loss
        )

        if self.enh_return_type == "waveform" and self.loss_type != "si_snr":
            # convert back to time-domain
            speech_pre = [
                self.enh_model.stft.inverse(ps, speech_lengths)[0] for ps in speech_pre
            ]

        if self.enh_return_type == "waveform":
            if resort_pre and perm != None:
                # resort the prediction wav with the perm from enh_loss
                # speech_pre : list[(bs,...)] of spk
                # perm : list[(num_spk)] of batch
                speech_pre_list = []
                for batch_idx, p in enumerate(perm):
                    batch_list = []
                    for spk_idx in p:
                        batch_list.append(speech_pre[spk_idx][batch_idx])  # spk,...
                    speech_pre_list.append(torch.stack(batch_list, dim=0))

                speech_pre = torch.stack(speech_pre_list, dim=0)  # bs,num_spk,...
            else:
                speech_pre = torch.stack(speech_pre, dim=1)  # bs,num_spk,...
        elif self.enh_return_type == "spectrum":
            # speech_pre: List([BS,T,F]) of complexTensor
            if resort_pre and perm != None:
                raise NotImplementedError("Resort with Spectrum not implemented.")

        return loss, perm, speech_pre, out_lengths

    def _calc_ctc_loss_with_spk(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        n_speakers: int = 1,
    ):
        # Calc CTC loss
        if n_speakers == 1:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            encoder_out, encoder_out_lens, ys_pad, ys_pad_lens = (
                torch.chunk(encoder_out, n_speakers, dim=0),
                torch.chunk(encoder_out_lens, n_speakers, dim=0),
                torch.chunk(ys_pad, n_speakers, dim=0),
                torch.chunk(ys_pad_lens, n_speakers, dim=0),
            )
            batch_size = encoder_out[0].size(0)
            loss_ctc = torch.stack(
                [
                    torch.stack(
                        [
                            self.ctc(
                                encoder_out[h],
                                encoder_out_lens[h],
                                ys_pad[r],
                                ys_pad_lens[r],
                            )
                            for r in range(n_speakers)
                        ],
                        dim=1,
                    )
                    for h in range(n_speakers)
                ],
                dim=2,
            )  # (B, n_ref, n_hyp)
            loss_ctc = loss_ctc.masked_fill(torch.isinf(loss_ctc), 0)
            perm_detail, min_loss_ctc = self.permutation_invariant_training(loss_ctc)
            loss_ctc = min_loss_ctc.mean()
            # permutate the encoder_out
            encoder_out, encoder_out_lens = (
                torch.stack(encoder_out, dim=1),
                torch.stack(encoder_out_lens, dim=1),
            )  # (B, n_spk, T, D)
            for b in range(batch_size):
                encoder_out[b] = encoder_out[b, perm_detail[b]]
                encoder_out_lens[b] = encoder_out_lens[b, perm_detail[b]]
            encoder_out = torch.cat(
                [encoder_out[:, i] for i in range(n_speakers)], dim=0
            )
            encoder_out_lens = torch.cat(
                [encoder_out_lens[:, i] for i in range(n_speakers)], dim=0
            )
            ys_pad = torch.cat(ys_pad, dim=0)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc, encoder_out, encoder_out_lens

    def _permutation_loss(self, ref, inf, criterion, perm=None):
        """The basic permutation loss function.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm: (batch)
        Returns:
            loss: torch.Tensor: (batch)
            perm: list[(num_spk)]
        """
        num_spk = len(ref)

        losses = torch.stack(
            [
                torch.stack([criterion(ref[r], inf[h]) for r in range(num_spk)], dim=1)
                for h in range(num_spk)
            ],
            dim=2,
        )  # (B, n_ref, n_hyp)
        perm_detail, min_loss = self.permutation_invariant_training(losses)

        return min_loss.mean(), perm_detail

    def permutation_invariant_training(self, losses: torch.Tensor):
        """Compute  PIT loss.

        Args:
            losses (torch.Tensor): (batch, nref, nhyp)
        Returns:
            perm: list: (batch, n_spk)
            loss: torch.Tensor: (batch)
        """
        hyp_perm, min_perm_loss = [], []
        losses_cpu = losses.data.cpu()
        for b, b_loss in enumerate(losses_cpu):
            # hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(b_loss)

            min_perm_loss.append(torch.mean(losses[b, row_ind, col_ind]))
            hyp_perm.append(col_ind)

        return hyp_perm, torch.stack(min_perm_loss)
