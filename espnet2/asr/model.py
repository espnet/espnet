from typing import Dict, List, Union
from typing import Tuple

import torch
from typeguard import typechecked

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import th_accuracy, make_pad_mask
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss \
    import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet2.utils.device_funcs import force_gatherable


class Model(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    @typechecked
    def __init__(self,
                 odim: int,
                 frontend: torch.nn.Module,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 ctc: CTC,
                 rnnt_decoder: torch.nn.Module = None,
                 ctc_weight: float = 0.5,
                 ignore_id: int = -1,
                 lsm_weight: float = 0.,
                 length_normalized_loss: bool = False,

                 report_cer: bool = False,
                 report_wer: bool = False,
                 char_list: Union[Tuple[str], List[str]] = (),
                 sym_space: str = '<space>',
                 sym_blank: str = '<blank>',
                 ):
        # TODO(kamo): Implement Interface class for frontend, encoder-decoder
        assert 0. < ctc_weight < 1., ctc_weight
        assert rnnt_decoder is None, 'Not implemented'

        char_list = tuple(char_list)
        
        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight

        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.rnnt_decoder = rnnt_decoder
        self.label_smoothing_loss = LabelSmoothingLoss(
            odim, ignore_id, lsm_weight, length_normalized_loss)

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                char_list, sym_space, sym_blank, report_cer, report_wer)
        else:
            self.error_calculator = None

    def forward(self,
                input: torch.Tensor,
                input_lengths: torch.Tensor,
                output: torch.Tensor,
                output_lengths: torch.Tensor)\
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """

        Args:
            input: (Batch, Length)
            input_lengths: (Batch, )
            output: (Batch, Length)
            output_lengths: (Batch,)
        """
        # TODO(kamo): Unify to either mask-way or length-way.
        # length-way may be better?

        assert input.dim() == 2, input.shape
        assert output.dim() == 2, output.shape
        assert input_lengths.dim() == 1, input_lengths.shape
        assert output_lengths.dim() == 1, output_lengths.shape
        assert (input.shape[0] == input_lengths.shape[0] ==
                output.shape[0] == output_lengths.shape[0]), \
            (input.shape,
             input_lengths.shape, output.shape, output_lengths.shape)
        batch_size = input.shape[0]

        # 0. Change pad_value
        # For data parallel
        input = input[:, :input_lengths.max()]
        output = output[:, :output_lengths.max()]
        input.masked_fill_(make_pad_mask(input_lengths).to(input.device), 0,)
        output.masked_fill_(make_pad_mask(output_lengths).to(output.device),
                            self.ignore_id)

        # 1. STFT, Feature transform
        # input (Batch, NSamples) -> input_feats: (Batch, Length, Dim)
        input_feats, feats_lens = self.frontend(input, input_lengths)

        # FIXME(kamo): Change the interface of Encoder-Decoder to use length-way
        feats_mask = (~make_pad_mask(feats_lens.tolist()))[:, None, :].to(
            feats_lens.device)

        # 2. Forward encoder
        # input_feats: (Batch, Length, Dim)
        encoder_out, encoder_out_mask = self.encoder(input_feats, feats_mask)

        # 3a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = \
                self._calc_decoder_loss(output, encoder_out, encoder_out_mask)

        # 3b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = \
                self._calc_ctc_loss(output, encoder_out, encoder_out_mask)

        # 3c. RNN-T branch (Is it possible?)
        if self.rnnt_decoder is not None:
            _ = \
                self._calc_rnnt_loss(output, encoder_out, encoder_out_mask)

        if self.ctc_weight == 0.:
            loss = loss_att
        elif self.ctc_weight == 1.:
            loss = loss_ctc
        else:
            loss = \
                self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

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
        loss, stats, weight = \
            force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_decoder_loss(self,
                           ys_pad: torch.Tensor,
                           encoder_out: torch.Tensor,
                           encoder_out_mask: torch.Tensor):
        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        # Forward decoder
        decoder_out, decoder_out_mask = \
            self.decoder(ys_in_pad, ys_mask, encoder_out, encoder_out_mask)

        # Compute attention loss
        loss_att = self.label_smoothing_loss(decoder_out, ys_out_pad)
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

    def _calc_ctc_loss(self,
                       ys_pad: torch.Tensor,
                       encoder_out: torch.Tensor,
                       encoder_out_mask: torch.Tensor
                       ):

        batch_size = encoder_out_mask.size(0)
        hs_len = encoder_out_mask.view(batch_size, -1).sum(1)
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, hs_len, ys_pad)

        # Calc CER using CTC
        cer_ctc = None
        if self.error_calculator is not None:
            ys_hat = \
                self.ctc.argmax(encoder_out_mask).data
            cer_ctc = \
                self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(self,
                        ys_pad: torch.Tensor, encoder_out: torch.Tensor,
                        encoder_out_mask: torch.Tensor):
        raise NotImplementedError
