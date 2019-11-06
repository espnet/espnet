from typing import Dict
from typing import Tuple

import torch
from pytypes import typechecked

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class E2E(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    @typechecked
    def __init__(self,
                 odim: int,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 ctc: CTC = None,
                 ctc_weight: float = 0.5,
                 ignore_id: int = -1,
                 lsm_weight: float = 0.,
                 length_normalized_loss: bool = True,

                 report_cer: bool = False,
                 report_wer: bool = False,
                 char_list: Tuple[str] = (),
                 sym_space: str = '<space>',
                 sym_blank: str = '<blank>',
                 ):
        super().__init__()
        assert 0. < ctc_weight < 1., ctc_weight
        self.odim = odim

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.label_smoothing_loss = LabelSmoothingLoss(
            odim, ignore_id, lsm_weight, length_normalized_loss)

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(char_list,
                                                    sym_space, sym_blank,
                                                    report_cer, report_wer)
        else:
            self.error_calculator = None

    def forward(self,
                xs_pad: torch.Tensor,
                ilens: torch.Tensor,
                ys_pad: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)

        # 1. Forward encoder
        encoder_out, encoder_out_mask = self.encoder(xs_pad, src_mask)

        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self.calc_decoder_loss(ys_pad, encoder_out, encoder_out_mask)

        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self.calc_ctc_loss(ys_pad, encoder_out, encoder_out_mask)

        if self.ctc_weight == 0.:
            loss = loss_att
        elif self.ctc_weight == 1.:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        stats = dict(loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
                     loss_att=loss_att.detach() if loss_att is not None else None,
                     acc=acc_att,
                     cer=cer_att,
                     wer=wer_att,
                     cer_ctc=cer_ctc,
                     loss=loss.detach())
        return loss, stats

    def calc_decoder_loss(self,
                          ys_pad: torch.Tensor,
                          encoder_out: torch.Tensor,
                          encoder_out_mask: torch.Tensor):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        # Forward decoder
        decoder_out, decoder_out_mask = self.decoder(
            ys_in_pad, ys_mask, encoder_out, encoder_out_mask)

        # Compute attention loss
        loss_att = self.label_smoothing_loss(decoder_out, decoder_out_mask)
        acc_att = th_accuracy(decoder_out.view(-1, self.odim), ys_out_pad,
                              ignore_label=self.ignore_id)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def calc_ctc_loss(self,
                      ys_pad: torch.Tensor,
                      encoder_out: torch.Tensor,
                      encoder_out_mask: torch.Tensor
                      ):
        batch_size = encoder_out_mask.size(0)
        hs_len = encoder_out_mask.view(batch_size, -1).sum(1)
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out.view(batch_size, -1, self.adim),
                            hs_len, ys_pad)

        # Calc CER using CTC
        cer_ctc = None
        if self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out_mask.view(batch_size, -1, self.adim)).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc
