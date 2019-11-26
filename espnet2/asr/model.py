from typing import Dict, List, Union, Optional
from typing import Tuple

import torch
from typeguard import typechecked

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy, make_pad_mask
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import \
    MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss \
    import LabelSmoothingLoss
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder_decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder_decoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.normalize.abs_normalization import AbsNormalization
from espnet2.train.abs_espnet_model import AbsESPNetModel
from espnet2.utils.device_funcs import force_gatherable


class ASRModel(AbsESPNetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    @typechecked
    def __init__(self,
                 vocab_size: int,
                 token_list: Union[Tuple[str, ...], List[str]],
                 frontend: Optional[AbsFrontend],
                 normalize: Optional[AbsNormalization],
                 encoder: AbsEncoder,
                 decoder: AbsDecoder,
                 ctc: CTC,
                 rnnt_decoder: torch.nn.Module = None,
                 ctc_weight: float = 0.5,
                 ignore_id: int = -1,
                 lsm_weight: float = 0.,
                 length_normalized_loss: bool = False,

                 report_cer: bool = True,
                 report_wer: bool = True,
                 sym_space: str = '<space>',
                 sym_blank: str = '<blank>',
                 ):
        assert 0. <= ctc_weight <= 1., ctc_weight
        assert rnnt_decoder is None, 'Not implemented'

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.normalize = normalize
        self.encoder = encoder
        self.decoder = decoder
        if ctc_weight == 0.:
            self.ctc = None
        else:
            self.ctc = ctc
        self.rnnt_decoder = rnnt_decoder
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss)

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer)
        else:
            self.error_calculator = None

    def forward(self,
                input: torch.Tensor,
                input_lengths: torch.Tensor,
                output: torch.Tensor,
                output_lengths: torch.Tensor)\
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            input: (Batch, Length, ...)
            input_lengths: (Batch, )
            output: (Batch, Length)
            output_lengths: (Batch,)
        """
        assert output_lengths.dim() == 1, output_lengths.shape
        # Check that batch_size is unified
        assert (input.shape[0] == input_lengths.shape[0] ==
                output.shape[0] == output_lengths.shape[0]), \
            (input.shape,
             input_lengths.shape, output.shape, output_lengths.shape)
        batch_size = input.shape[0]

        # 0. Change pad_value
        output = output[:, :output_lengths.max()]  # for data-parallel
        output.masked_fill_(make_pad_mask(output_lengths, output, 1),
                            self.ignore_id)

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(input, input_lengths)

        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, output, output_lengths)

        # 2b. CTC branch
        if self.ctc_weight == 0.:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, output, output_lengths)

        # 2c. RNN-T branch (Is it possible?)
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(
                encoder_out, encoder_out_lens, output, output_lengths)

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

    def encode(self, input: torch.Tensor, input_lengths: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_recog.py

        Args:
            input: (Batch, Length, ...)
            input_lengths: (Batch, )
        """
        assert input_lengths.dim() == 1, input_lengths.shape

        # 0. Change pad_value
        input = input[:, :input_lengths.max()]  # for data-parallel
        input.masked_fill_(make_pad_mask(input_lengths, input, 1), 0,)

        if self.frontend is not None:
            # 1. Frontend
            #  e.g. STFT and Feature transform
            #       data_loader may send time-domain signal in this case
            # input (Batch, NSamples) -> input_feats: (Batch, NFrames, Dim)
            input_feats, feats_lens = self.frontend(input, input_lengths)
        else:
            # 1. No frontend:
            #  data_loader must send feature in this case
            input_feats, feats_lens = input, input_lengths

        # 2. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            input_feats, feats_lens = self.normalize(input_feats, feats_lens)

        # 3. Forward encoder
        # input_feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = \
            self.encoder(input_feats, feats_lens)

        assert encoder_out.size(0) == input.size(0), \
            (encoder_out.size(), input.size(0))
        assert encoder_out.size(1) <= encoder_out_lens.max(), \
            (encoder_out.size(), encoder_out_lens.max())

        return encoder_out, encoder_out_lens

    def _calc_att_loss(self,
                       encoder_out: torch.Tensor,
                       encoder_out_lens: torch.Tensor,
                       ys_pad: torch.Tensor,
                       ys_pad_lens: torch.Tensor,
                       ):
        ys_in_pad, ys_out_pad = \
            add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = \
            self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(decoder_out.view(-1, self.vocab_size),
                              ys_out_pad, ignore_label=self.ignore_id)

        # Compute cer/wer using attention-decoder
        if self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = \
                self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(self,
                       encoder_out: torch.Tensor,
                       encoder_out_lens: torch.Tensor,
                       ys_pad: torch.Tensor,
                       ys_pad_lens: torch.Tensor,
                       ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = \
                self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(self,
                        encoder_out: torch.Tensor,
                        encoder_out_mask: torch.Tensor,
                        ys_pad: torch.Tensor,
                        ys_pad_lens: torch.Tensor,
                        ):
        raise NotImplementedError
