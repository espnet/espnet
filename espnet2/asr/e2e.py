from typing import Dict, List, Union
from typing import Tuple

import torch
from typeguard import typechecked

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss \
    import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class E2E(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    @typechecked
    def __init__(self,
                 odim: int,
                 stft: torch.nn.Module,
                 frontend: torch.nn.Module,
                 feature_transform: torch.nn.Module,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 ctc: torch.nn.Module,
                 ctc_weight: float = 0.5,
                 ignore_id: int = -1,
                 lsm_weight: float = 0.,
                 length_normalized_loss: bool = True,

                 report_cer: bool = False,
                 report_wer: bool = False,
                 char_list: Union[Tuple[str], List[str]] = (),
                 sym_space: str = '<space>',
                 sym_blank: str = '<blank>',
                 ):
        assert 0. < ctc_weight < 1., ctc_weight
        super().__init__()
        self.odim = odim
        self.ingore_id = ignore_id

        self.stft = stft
        self.frontend = frontend
        self.feature_transform = feature_transform
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
                input: torch.Tensor,
                input_mask: torch.Tensor,
                output: torch.Tensor,
                output_mask: torch.Tensor,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """

        Args:
            input: (Batch, Length, Dim)
            input_mask: (Batch,)
            output: (Batch, Length)
            output_mask: (Batch,)
        """
        # 0. Change pad_value
        input.masked_fill_(input_mask, 0,)
        output.masked_fill_(output_mask, self.ignore_id)

        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft = self.stft(input)

        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            input_stft, hlens, mask = self.frontend(input_stft, ilens)

        # 3. Feature transform e.g. Stft -> Mel-Fbank
        input_feats, hlens = self.feature_transform(input_stft, hlens)

        # 4. Forward encoder
        encoder_out, encoder_out_mask = self.encoder(input_feats, input_mask)

        # 5a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self.calc_decoder_loss(
                output, encoder_out, encoder_out_mask)

        # 6b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self.calc_ctc_loss(
                output, encoder_out, encoder_out_mask)

        if self.ctc_weight == 0.:
            loss = loss_att
        elif self.ctc_weight == 1.:
            loss = loss_ctc
        else:
            loss = \
                self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        stats = dict(loss_ctc=
                     loss_ctc.detach() if loss_ctc is not None else None,
                     loss_att=
                     loss_att.detach() if loss_att is not None else None,
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
        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.sos, self.eos, self.ignore_id)
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
            cer_att, wer_att = self.error_calculator(
                ys_hat.cpu(), ys_pad.cpu())

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
            ys_hat = self.ctc.argmax(
                encoder_out_mask.view(batch_size, -1, self.adim)).data
            cer_ctc = self.error_calculator(
                ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc
