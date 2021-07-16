from contextlib import contextmanager
from distutils.version import LooseVersion
from itertools import groupby
import logging
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy
import torch
from typeguard import check_argument_types

from espnet.nets.beam_search import Hypothesis
from espnet.nets.pytorch_backend.maskctc.add_mask_token import mask_uniform
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.masked_lm_decoder import MaskedLMDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.text.token_id_converter import TokenIDConverter


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class MaskCTCModel(ESPnetASRModel):
    """Mask-CTC-based non-autoregressive model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
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
        assert 0.0 < ctc_weight < 1.0, ctc_weight
        assert isinstance(
            decoder, MaskedLMDecoder
        ), "Only masked_lm decoder is supported"
        assert rnnt_decoder is None, "Not implemented"

        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            rnnt_decoder=rnnt_decoder,
            ctc_weight=ctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
        )

        # add <mask> and override some fields
        token_list.append("<mask>")
        vocab_size += 1
        self.mask_token = vocab_size - 1
        self.sos = vocab_size - 2
        self.eos = vocab_size - 2

        self.vocab_size = vocab_size
        self.token_list = token_list.copy()

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # 0. Mask tokens
        ys_in_pad, ys_out_pad = mask_uniform(
            ys_pad, self.mask_token, self.eos, self.ignore_id
        )

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_pad_lens
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


class MaskCTCInference(torch.nn.Module):
    """Mask-CTC-based non-autoregressive inference"""

    def __init__(
        self,
        asr_model: MaskCTCModel,
        n_iterations: int,
        threshold_probability: float,
    ):
        """Initialize Mask-CTC inference"""
        super().__init__()
        self.ctc = asr_model.ctc
        self.mlm = asr_model.decoder
        self.mask_token = asr_model.mask_token
        self.n_iterations = n_iterations
        self.threshold_probability = threshold_probability
        self.converter = TokenIDConverter(token_list=asr_model.token_list)

    def ids2text(self, ids: List[int]):
        text = "".join(self.converter.ids2tokens(ids))
        return text.replace("<mask>", "_").replace("<space>", " ")

    def forward(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Perform Mask-CTC inference"""
        # greedy ctc outputs
        enc_out = enc_out.unsqueeze(0)
        ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(enc_out)).max(dim=-1)
        y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

        # calculate token-level ctc probabilities by taking
        # the maximum probability of consecutive frames with
        # the same ctc symbols
        probs_hat = []
        cnt = 0
        for i, y in enumerate(y_hat.tolist()):
            probs_hat.append(-1)
            while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                if probs_hat[i] < ctc_probs[0][cnt]:
                    probs_hat[i] = ctc_probs[0][cnt].item()
                cnt += 1
        probs_hat = torch.from_numpy(numpy.array(probs_hat))

        # mask ctc outputs based on ctc probabilities
        p_thres = self.threshold_probability
        mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
        confident_idx = torch.nonzero(probs_hat[y_idx] >= p_thres).squeeze(-1)
        mask_num = len(mask_idx)

        y_in = torch.zeros(1, len(y_idx), dtype=torch.long) + self.mask_token
        y_in[0][confident_idx] = y_hat[y_idx][confident_idx]

        logging.info("ctc:{}".format(self.ids2text(y_in[0].tolist())))

        # iterative decoding
        if not mask_num == 0:
            K = self.n_iterations
            num_iter = K if mask_num >= K and K > 0 else mask_num

            for t in range(num_iter - 1):
                pred, _ = self.mlm(enc_out, [enc_out.size(1)], y_in, [y_in.size(1)])
                pred_score, pred_id = pred[0][mask_idx].max(dim=-1)
                cand = torch.topk(pred_score, mask_num // num_iter, -1)[1]
                y_in[0][mask_idx[cand]] = pred_id[cand]
                mask_idx = torch.nonzero(y_in[0] == self.mask_token).squeeze(-1)

                logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

            # predict leftover masks (|masks| < mask_num // num_iter)
            pred, _ = self.mlm(enc_out, [enc_out.size(1)], y_in, [y_in.size(1)])
            y_in[0][mask_idx] = pred[0][mask_idx].argmax(dim=-1)

            logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        yseq = torch.tensor(
            [self.mask_token] + y_in.tolist()[0] + [self.mask_token], device=y_in.device
        )

        return [Hypothesis(yseq=yseq)]
