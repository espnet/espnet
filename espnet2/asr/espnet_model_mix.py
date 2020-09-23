from contextlib import contextmanager
from distutils.version import LooseVersion
from itertools import chain
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.e2e_asr_mix import PIT
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
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRMixModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder multi-speaker model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
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
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.num_spkrs = encoder.num_spkrs
        self.pit = PIT(self.num_spkrs)

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

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        # text: torch.Tensor,
        # text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + EncoderMix + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        # transcript of each speaker
        text_ref = [
            kwargs["text_ref{}".format(spk + 1)] for spk in range(self.num_spkrs)
        ]
        text_ref_lengths = [
            kwargs["text_ref{}_lengths".format(spk + 1)]
            for spk in range(self.num_spkrs)
        ]
        assert all(txt_length.dim() == 1 for txt_length in text_ref_lengths), (
            txt_length.shape for txt_length in text_ref_lengths
        )
        # Check that batch_size is unified
        batch_size = speech.shape[0]
        assert batch_size == speech_lengths.shape[0], (
            speech.shape,
            speech_lengths.shape,
        )
        assert all(
            it.shape[0] == batch_size for it in chain(text_ref, text_ref_lengths)
        ), (
            speech.shape,
            (txt.shape for txt in text_ref),
            (txt_length.shape for txt_length in text_ref_lengths),
        )

        # for data-parallel
        text_length_max = max(txt_length.max() for txt_length in text_ref_lengths)
        # num_spkrs * Tensor[B, text_length_max]
        text_ref = [
            torch.cat(
                [
                    txt,
                    torch.ones(batch_size, text_length_max, dtype=txt.dtype).to(
                        txt.device
                    )
                    * self.idx_blank,
                ],
                dim=1,
            )[:, :text_length_max]
            for txt in text_ref
        ]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2a. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc, min_perm = None, None, None
        else:
            loss_ctc, cer_ctc, min_perm = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text_ref, text_ref_lengths, perm=None
            )

        # 2b. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att, min_perm = self._calc_att_loss(
                encoder_out, encoder_out_lens, text_ref, text_ref_lengths, perm=min_perm
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(
                encoder_out, encoder_out_lens, text_ref, text_ref_lengths, perm=min_perm
            )

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
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
        # -> encoder_out: List[Tensor(Batch, Length2, Dim2)]
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert isinstance(encoder_out, list), type(encoder_out)
        batchsize = speech.size(0)
        for i, enc_out in enumerate(encoder_out):
            assert enc_out.size(0) == batchsize, (enc_out.size(), batchsize)
            assert enc_out.size(1) <= encoder_out_lens[i].max(), (
                enc_out.size(),
                encoder_out_lens[i].max(),
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
        encoder_out: List[torch.Tensor],
        encoder_out_lens: List[torch.Tensor],
        ys_pad: List[torch.Tensor],
        ys_pad_lens: List[torch.Tensor],
        perm=None,
    ):
        if perm is None:
            raise NotImplementedError
        else:
            min_perm = perm
            # (num_spkrs, B, Lmax)
            ys_pad_new = torch.stack(ys_pad, dim=0)
            ys_pad_lens_new = torch.stack(ys_pad_lens, dim=0)
            for i in range(ys_pad_new.size(1)):  # B
                ys_pad_new[:, i] = ys_pad_new[min_perm[i], i]
                ys_pad_lens_new[:, i] = ys_pad_lens_new[min_perm[i], i]

        acc_att = []
        decoder_out = []
        loss_att = []
        for spk in range(self.num_spkrs):
            ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad_new[spk], self.sos, self.eos, self.ignore_id
            )
            ys_in_lens = ys_pad_lens_new[spk] + 1

            # 1. Forward decoder
            dec_out, _ = self.decoder(
                encoder_out[spk], encoder_out_lens[spk], ys_in_pad, ys_in_lens
            )
            decoder_out.append(dec_out)

            # 2. Compute attention loss
            loss_att.append(self.criterion_att(dec_out, ys_out_pad))
            acc_att.append(
                th_accuracy(
                    dec_out.view(-1, self.vocab_size),
                    ys_out_pad,
                    ignore_label=self.ignore_id,
                )
            )
        loss_att = torch.mean(loss_att)
        acc_att = np.mean(acc_att)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            cer_att, wer_att = [], []
            for spk in range(self.num_spkrs):
                ys_hat = decoder_out[spk].argmax(dim=-1)
                cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad_new[spk].cpu())
                cer_att.append(cer)
                wer_att.append(wer)
            cer_att = np.mean(cer_att)
            wer_att = np.mean(wer_att)

        return loss_att, acc_att, cer_att, wer_att, min_perm

    def _calc_ctc_loss(
        self,
        encoder_out: List[torch.Tensor],
        encoder_out_lens: List[torch.Tensor],
        ys_pad: List[torch.Tensor],
        ys_pad_lens: List[torch.Tensor],
        perm=None,
    ):
        # Calc CTC loss
        if perm is None:
            # (B, num_spkrs ** 2)
            loss_ctc_perm = torch.stack(
                [
                    self.ctc(
                        encoder_out[i // self.num_spkrs],
                        encoder_out_lens[i // self.num_spkrs],
                        ys_pad[i % self.num_spkrs],
                        ys_pad_lens[i % self.num_spkrs],
                    )
                    for i in range(self.num_spkrs ** 2)
                ],
                dim=1,
            )
            loss_ctc, min_perm = self.pit.pit_process(loss_ctc_perm)

            # (num_spkrs, B, Lmax)
            ys_pad_new = torch.stack(ys_pad, dim=0)
            for i in range(ys_pad_new.size(1)):  # B
                ys_pad_new[:, i] = ys_pad_new[min_perm[i], i]
        else:
            min_perm = perm
            # (num_spkrs, B, Lmax)
            ys_pad_new = torch.stack(ys_pad, dim=0)
            ys_pad_lens_new = torch.stack(ys_pad_lens, dim=0)
            for i in range(ys_pad_new.size(1)):  # B
                ys_pad_new[:, i] = ys_pad_new[min_perm[i], i]
                ys_pad_lens_new[:, i] = ys_pad_lens_new[min_perm[i], i]
            loss_ctc = torch.mean(
                [
                    self.ctc(
                        encoder_out[i],
                        encoder_out_lens[i],
                        ys_pad_new[i],
                        ys_pad_lens_new[i],
                    )
                    for i in range(self.num_spkrs)
                ]
            )

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = [self.ctc.argmax(enc_out).data for enc_out in encoder_out]
            cer_ctc = np.mean(
                [
                    self.error_calculator(
                        yhat.cpu(), ys_pad_new[spk].cpu(), is_ctc=True
                    )
                    for spk, yhat in enumerate(ys_hat)
                ]
            )

        return loss_ctc, cer_ctc, min_perm

    def _calc_rnnt_loss(
        self,
        encoder_out: List[torch.Tensor],
        encoder_out_lens: List[torch.Tensor],
        ys_pad: List[torch.Tensor],
        ys_pad_lens: List[torch.Tensor],
        perm=None,
    ):
        raise NotImplementedError
