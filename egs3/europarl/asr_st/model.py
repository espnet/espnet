
import torch
import torch.nn as nn

from espnet2.bin.s2t_inference import Speech2Text
from espnet.nets.pytorch_backend.nets_utils import (
    th_accuracy, pad_list
)
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss


class OWSMFinetune(nn.Module):
    def __init__(self, model_tag):
        super().__init__()
        owsm_model = Speech2Text.from_pretrained(model_tag)
        self.model = owsm_model.s2t_model
        self.tokenizer = owsm_model.tokenizer
        self.converter = owsm_model.converter

        self.na = owsm_model.s2t_model.na
        self.sos = owsm_model.s2t_model.sos
        self.eos = owsm_model.s2t_model.eos
        self.sop = owsm_model.s2t_model.sop
        self.ignore_id = owsm_model.s2t_model.ignore_id

        self.ctc_weight = owsm_model.s2t_model.ctc_weight

        self.criterion_att = LabelSmoothingLoss(
            size=owsm_model.s2t_model.vocab_size, # vocab size
            padding_idx=-1,
            smoothing=0.1,
            normalize_length=False,
        )

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Filter out invalid samples where text is not available
        is_valid = [self.na not in y for y in ys_pad]
        if not any(is_valid):
            return torch.tensor(0.0), None  

        encoder_out = encoder_out[is_valid]
        encoder_out_lens = encoder_out_lens[is_valid]
        ys_pad = ys_pad[is_valid]
        ys_pad_lens = ys_pad_lens[is_valid]

        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        return loss_ctc

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        ys_prev_pad: torch.Tensor,
        ys_prev_lens: torch.Tensor,
    ):
        # 0. Prepare input and output with sos, eos, sop
        ys = [y[y != self.ignore_id] for y in ys_pad]
        ys_prev = [y[y != self.ignore_id] for y in ys_prev_pad]

        _sos = ys_pad.new([self.sos])
        _eos = ys_pad.new([self.eos])
        _sop = ys_pad.new([self.sop])
        ys_in = []
        ys_in_lens = []
        ys_out = []
        for y_prev, y in zip(ys_prev, ys):
            if self.na in y_prev:
                # Prev is not available in this case
                y_in = [_sos, y]
                y_in_len = len(y) + 1
                y_out = [y, _eos]
            else:
                y_in = [_sop, y_prev, _sos, y]
                y_in_len = len(y_prev) + len(y) + 2
                y_out = [self.ignore_id * ys_pad.new_ones(len(y_prev) + 1), y, _eos]

            ys_in.append(torch.cat(y_in))
            ys_in_lens.append(y_in_len)
            ys_out.append(torch.cat(y_out))

        ys_in_pad = pad_list(ys_in, self.eos)
        ys_in_lens = torch.tensor(ys_in_lens).to(ys_pad_lens)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

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
        return loss_att, acc_att
    
    def forward(
            self, speech, speech_lengths,
            text, text_lengths,
            text_ctc, text_ctc_lengths,
            text_prev, text_prev_lengths,
        ):
        stats = dict()
        encoder_out, encoder_out_lens = self.model.encode(
            speech, speech_lengths
        )

        # OWSM has ctc.
        loss_ctc = self._calc_ctc_loss(
            encoder_out, encoder_out_lens, text_ctc, text_ctc_lengths
        )

        # calculate attention loss
        loss_att, acc_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, 
            text, text_lengths,
            text_prev, text_prev_lengths,
        )

        if self.ctc_weight > 0.0:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        else:
            loss = loss_att

        stats['loss_att'] = loss_att.detach()
        stats['loss_ctc'] = loss_ctc.detach()
        stats['loss'] = loss.detach()
        stats['acc_att'] = acc_att.detach()

        return loss, stats, None

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        return {"feats": speech, "feats_lengths": speech_lengths}
