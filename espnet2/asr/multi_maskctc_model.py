import logging
from contextlib import contextmanager
from itertools import groupby
import re
from typing import Dict, List, Optional, Tuple, Union

from librosa import ex
import numpy
import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.multi_mask_mlm_decoder import MultiMaskMLMDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.legacy.nets.beam_search import Hypothesis
from espnet2.legacy.nets.e2e_asr_common import ErrorCalculator
from espnet2.legacy.nets.pytorch_backend.maskctc.add_mask_token import apply_mask
from espnet2.legacy.nets.pytorch_backend.nets_utils import th_accuracy
from espnet2.legacy.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,
)
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.legacy.nets.pytorch_backend.nets_utils import pad_list

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

import torch.nn.functional as F

class MultiMaskCTCModel(ESPnetASRModel):
    """Hybrid CTC/Masked LM Encoder-Decoder model (Mask-CTC)"""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: MultiMaskMLMDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module] = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        sym_mask: str = "<mask>",
        extract_feats_in_collect_stats: bool = True,
        num_hypotheses: int = 1,
    ):

        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
        )

        # Add <mask> and override inherited fields
        token_list.append(sym_mask)
        vocab_size += 1
        self.vocab_size = vocab_size
        self.mask_token = vocab_size - 1
        self.token_list = token_list.copy()
        self.num_hypotheses = num_hypotheses
        # MLM loss
        del self.criterion_att
        self.criterion_mlm = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.error_calculator = None
        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # For data-parallel
        text = text[:, : text_lengths.max()]

        # Define stats to report
        loss_mlm, acc_mlm = None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # 2. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # 2a. Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        # 3. MLM decoder branch
        if self.ctc_weight != 1.0:
            loss_mlm, acc_mlm = self._calc_mlm_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 4. CTC/MLM loss definition
        if self.ctc_weight == 0.0:
            loss = loss_mlm
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_mlm

        # Collect MLM branch stats
        stats["loss_mlm"] = loss_mlm.detach() if loss_mlm is not None else None
        stats["acc_mlm"] = acc_mlm

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_mlm_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # 1. Apply masks
        ys_in_pad, ys_out_pad = apply_mask(
                ys_pad, self.mask_token, 
                self.eos, self.ignore_id,
                self.num_hypotheses
            )
        

        # 2. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_pad_lens
        )

        # 3. Compute mlm loss
        loss_mlm = self.criterion_mlm(decoder_out, ys_out_pad)
        acc_mlm = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        return loss_mlm, acc_mlm

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        raise NotImplementedError


class Multi_MaskCTCInference(torch.nn.Module):
    """Mask-CTC-based non-autoregressive inference"""

    def __init__(
        self,
        asr_model: MultiMaskCTCModel,
        n_iterations: int,
        threshold_probability: float,
        num_hypotheses: int = 10,
        update_type: int = 1,
    ):
        """Initialize Mask-CTC inference"""
        super().__init__()
        self.ctc = asr_model.ctc
        self.mlm = asr_model.decoder
        self.eos = asr_model.eos
        self.mask_token = asr_model.mask_token
        self.n_iterations = n_iterations
        self.threshold_probability = threshold_probability
        self.converter = TokenIDConverter(token_list=asr_model.token_list)
        self.num_hypotheses = 5
        self.max_score = 0.0
        self.update_type = update_type

    def ids2text(self, ids: List[int]):
        text = "".join(self.converter.ids2tokens(ids))
        return text.replace("<mask>", "_").replace("<space>", " ")


    def compress_and_mean(self, A, B):
        # A, B: (T,) の 1次元テンソル
        # 連続区間の開始位置を検出
        start = torch.ones_like(A, dtype=torch.bool)
        start[1:] = A[1:] != A[:-1]

        # 各要素が属する「連続区間ID」を作る
        group_id = torch.cumsum(start, dim=0) - 1   # 0,0,1,1,1,2,3,3 のようなID

        # ユニークな A（連続区間ごと）
        A_unique = A[start]

        # 各 group_id ごとに B の平均を取る
        # scatter_add を使って高速に集計
        num_groups = A_unique.size(0)
        sum_B = torch.zeros(num_groups, dtype=B.dtype)
        cnt_B = torch.zeros(num_groups, dtype=B.dtype)

        sum_B.scatter_add_(0, group_id, B)
        cnt_B.scatter_add_(0, group_id, torch.ones_like(B, dtype=B.dtype))

        B_mean = sum_B / cnt_B

        return A_unique, B_mean

    
    
    def top_k_sample(self, logits, k):
        # logits: (V,)
        # 1. 上位 k の値とインデックスを取得
        #breakpoint()
        topk_vals, topk_idx = torch.topk(logits, k)

        # 2. 上位 k の logits だけで softmax
        probs = torch.softmax(topk_vals, dim=-1)

        # 3. その分布からサンプリング
        sampled_idx = torch.multinomial(probs[0], 1)
        
        # 4. 元の語彙 ID に戻す
        idx = torch.gather(topk_idx, -1, sampled_idx.unsqueeze(-1).transpose(0,1)).squeeze(-1).squeeze(0)
        probs = probs.max(-1)[0].squeeze(0)
        nonzero_mask = idx != 0
        if nonzero_mask.sum() == 0:
            return self.top_k_sample(logits, k)
        return self.compress_and_mean(idx[nonzero_mask].cpu(), probs[nonzero_mask].cpu())

    def add_gumbel_noise(self, logits, temperature):
        '''
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        '''
        #breakpoint()
        if temperature == 0:
            return logits
        #logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=logits.dtype)
        gumbel_noise = (- torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def forward(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        enc_out = enc_out.unsqueeze(0)
        ctc_logit = self.ctc.logit(enc_out) #torch.exp(self.ctc.log_softmax(enc_out))
        # ctc_logits, probs = [], []
        # for i in range(self.num_hypotheses):
        #     _logit, prob = self.top_k_sample(ctc_logit, self.num_hypotheses)
        #     ctc_logits.append(_logit)
        #     probs.append(prob)

        # y_in = pad_list(ctc_logits, self.eos).cuda()
        # probs = pad_list(probs, 1.0).cuda()
        # #breakpoint()
        # y_in[probs<self.threshold_probability]  = self.mask_token
        # y_in_length = (y_in != self.eos).sum(-1)
        # enc_length = torch.LongTensor([enc_out.size(1)])
        # if self.n_iterations > 0:
        #     for t in range(self.n_iterations - 1):
        #         pred, _ = self.mlm(enc_out, enc_length, y_in, y_in_length)
        #         if pred.size(0) > self.num_hypotheses:
        #             rerank_idx = torch.topk(pred.max(-1)[0].mean(1), self.num_hypotheses)[1]
        #             pred = pred[rerank_idx]
        #         logits, probs = [], []
        #         for i in range(self.num_hypotheses):
        #             #breakpoint()
        #             for j in range(pred.size(0)):
        #                 _logit, prob = self.top_k_sample(pred[j].unsqueeze(0), self.num_hypotheses)
        #                 logging.info("msk:{}".format(self.ids2text(_logit.tolist())))
        #                 logits.append(_logit)
        #                 probs.append(prob)
        #         y_in = pad_list(logits, self.eos).cuda()
        #         probs = pad_list(probs, 1.0).cuda()
                
        #         #y_in[ probs < self.threshold_probability]  = self.mask
        #         y_in_length = (y_in != self.eos).sum(-1)
        #         #breakpoint()
             
        #ctc_logits = 
        ctc_ids = ctc_logit.argmax(dim=-1)
        y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

        logging.info("ctc:{}".format(self.ids2text(y_hat[y_idx].tolist())))
        y_in = y_hat[y_idx]
        if len(y_in) == 0:
            y_in = torch.tensor([self.mask_token], device=enc_out.device).unsqueeze(0).repeat(self.num_hypotheses, 1)
            mask_idx = torch.tensor([True], device=enc_out.device).unsqueeze(0).repeat(self.num_hypotheses, 1)
            ctc_hat = torch.zeros((y_in.size(0), self.mask_token+1), device=enc_out.device)
        else:
            # calculate token-level ctc probabilities by taking
            # the maximum probability of consecutive frames with
            # the same ctc symbols
            ctc_hat = []
            #logit_prob
            cnt = 0
            for i, y in enumerate(y_hat.tolist()):
                ctc_hat.append(torch.tensor([-1]).cuda())
                while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                    #breakpoint()
                    if ctc_hat[i].max() < ctc_logit[0][cnt].max():
                        ctc_hat[i] = ctc_logit[0][cnt] #.item()
                        #logit_prob
                    cnt += 1
            #breakpoint()
            ctc_hat = torch.stack(ctc_hat)[y_idx]
            ctc_hat[:, 0] = -1  #  blank id
            #breakpoint()
            ctc_hat = torch.cat((ctc_hat, torch.zeros((ctc_hat.size(0), 1), device=ctc_hat.device)), dim=-1)  # add mask token prob
            mask_idx = ctc_hat.max(-1)[0] < self.threshold_probability
            mask_num = mask_idx.sum()
            y_in, mask_idx = self.generate_hypothesis(y_in, ctc_hat, mask_idx,)

        # for i in range(y_in.size(0)):
        #     logging.info("hypo:{}".format(self.ids2text(y_in[i].tolist())))
        enc_length = torch.LongTensor([enc_out.size(1)])
        y_in_length = torch.LongTensor([y_in.size(1)])
        # iterative decoding
        ctc_hat = F.normalize(ctc_hat, p=2, dim=1)
        if self.n_iterations > 0:
            for t in range(self.n_iterations - 1):
                pred, _ = self.mlm(enc_out, enc_length, y_in, y_in_length)
                
                pred = F.normalize(pred, p=2, dim=1)
                prob = self.add_gumbel_noise(pred+ctc_hat*0.1, 1)
                # Re-rank
                if prob.size(0) > self.num_hypotheses:
                    rerank_idx = torch.topk(prob.max(-1)[0].mean(1), self.num_hypotheses)[1]
                    prob = prob[rerank_idx]
                    y_in = y_in[rerank_idx]
                    mask_idx = mask_idx[rerank_idx]

                new_y_in = []
                new_mask_idxs = []
                for i in range(prob.size(0)):
                    _y_in, _mask_idx = self.update_hypothesis(y_in[i], prob[i], mask_idx[0])
                    logging.info("msk:{}".format(self.ids2text(_y_in.tolist())))
                    _y_in, _mask_idx = self.generate_hypothesis(_y_in, prob[i], _mask_idx)
                    new_y_in.append(_y_in)
                    new_mask_idxs.append(_mask_idx)
                y_in = torch.cat(new_y_in, dim=0)
                mask_idx = torch.cat(new_mask_idxs, dim=0)
                #breakpoint()
                
                
                #breakpoint()
            # predict leftover masks (|masks| < mask_num // num_iter)
            pred, _ = self.mlm(enc_out, enc_length, y_in, y_in_length)
            prob = pred.softmax(dim=-1)
            # Re-rank
            if prob.size(0) > self.num_hypotheses:
                prob = prob[torch.topk(prob.max(-1)[0].mean(1), self.num_hypotheses)[1]]
            y_in, _ = self.update_hypothesis(y_in[0], prob[0], mask_idx[0])
            y_in = y_in.unsqueeze(0)      
        else:
            y_in = y_hat[y_idx].unsqueeze(0)
        # pad with mask tokens to ensure compatibility with sos/eos tokens
        yseq = torch.tensor(
            [self.mask_token] + y_in.tolist()[0] + [self.mask_token], device=y_in.device
        )
        
        return Hypothesis(yseq=yseq)
    

    def _foward_type1(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Perform Mask-CTC inference"""
        # greedy ctc outputs
        enc_out = enc_out.unsqueeze(0)
        ctc_logit, ctc_ids = torch.exp(self.ctc.log_softmax(enc_out)).max(dim=-1)
        y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

        logging.info("ctc:{}".format(self.ids2text(y_hat[y_idx].tolist())))

        # calculate token-level ctc probabilities by taking
        # the maximum probability of consecutive frames with
        # the same ctc symbols
        ctc_hat = []
        cnt = 0
        for i, y in enumerate(y_hat.tolist()):
            ctc_hat.append(-1)
            while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                if ctc_hat[i] < ctc_logit[0][cnt]:
                    ctc_hat[i] = ctc_logit[0][cnt].item()
                cnt += 1
        ctc_hat = torch.from_numpy(numpy.array(ctc_hat)).to(enc_out.device)

        # mask ctc outputs based on ctc probabilities
        p_thres = self.threshold_probability
        mask_idx = torch.nonzero(ctc_hat[y_idx] < p_thres).squeeze(-1)

        confident_idx = torch.nonzero(ctc_hat[y_idx] >= p_thres).squeeze(-1)
        mask_num = len(mask_idx)

        y_in = (
            torch.zeros(1, len(y_idx), dtype=torch.long).to(enc_out.device)
            + self.mask_token
        )
        y_in[0][confident_idx] = y_hat[y_idx][confident_idx]

        logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))
        enc_length = torch.LongTensor([enc_out.size(1)])
        y_in_length = torch.LongTensor([y_in.size(1)])
        # iterative decoding
        if not mask_num == 0:
            K = self.n_iterations
            num_iter = K if mask_num >= K and K > 0 else mask_num

            for t in range(num_iter - 1):
                pred, _ = self.mlm(enc_out, enc_length, y_in, y_in_length)
                prob = pred.softmax(dim=-1)
                # Re-rank
                if prob.size(0) > self.num_hypotheses:
                    prob = prob[torch.topk(prob.max(-1)[0].mean(1), self.num_hypotheses)[1]]
                new_y_in = []
                new_mask_idxs = []
                for i in range(pred.size(0)):
                    _y_in, _mask_idx = self.generate_hypothesis(y_in[i], prob[i], mask_idx[0])
                    new_y_in.append(_y_in)
                    new_mask_idxs.append(_mask_idx)
                y_in = torch.stack(new_y_in)
                mask_idx = torch.stack(new_mask_idxs)[0]
                logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

            # predict leftover masks (|masks| < mask_num // num_iter)
            pred, _ = self.mlm(enc_out, enc_length, y_in, y_in_length)
            prob = pred.softmax(dim=-1)
            # Re-rank
            if prob.size(0) > self.num_hypotheses:
                prob = prob[torch.topk(prob.max(-1)[0].mean(1), self.num_hypotheses)[1]]
            y_in, _ = self.generate_hypothesis(y_in[0], prob[0], mask_idx[0])
            y_in = y_in.unsqueeze(0)      
            logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        yseq = torch.tensor(
            [self.mask_token] + y_in.tolist()[0] + [self.mask_token], device=y_in.device
        )

        return Hypothesis(yseq=yseq)

    def generate_hypothesis(self, y_in, logit_prob, mask_idx, return_mask=False):
        try:
            if return_mask:
                y_in = y_in.unsqueeze(0).repeat(self.num_hypotheses, 1)
                mask_idx = mask_idx.unsqueeze(0).repeat(self.num_hypotheses, 1)
                y_in[:] = self.mask_token
                mask_idx[:] = True
            else:
                y_in = y_in.unsqueeze(0).repeat(self.num_hypotheses, 1)
                mask_idx = mask_idx.unsqueeze(0).repeat(self.num_hypotheses, 1)
                hypotheses = logit_prob.topk(
                        self.num_hypotheses, -1)[1].transpose(0, 1)
                y_in[1:][mask_idx[1:]] = hypotheses[:-1][mask_idx[1:]]
                y_in[-1][mask_idx[-1]] = self.mask_token    
        except:
            pass
        return y_in, mask_idx

    def update_hypothesis(self,  y_in, logit_prob, mask_idx):
        # update hypothesis based on pred
        try:
            y_in[mask_idx] = self.mask_token
        except:
            pass
        pred_score, pred_id = logit_prob.max(dim=-1)
        try:
            cand = torch.topk(
                    pred_score[~mask_idx], 1, -1)[1]
            y_in[~mask_idx][cand] = pred_id[~mask_idx][cand]
        except:
            pass
        try:
            # update mask
            cand = torch.topk(
                    pred_score[mask_idx], mask_idx.sum() // self.n_iterations, -1)[1]
            y_in[mask_idx][cand] = pred_id[mask_idx][cand]
        
            mask_idx[cand] = False
        except:
            pass
        return y_in, mask_idx
    
    def _foward_type2(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Perform Mask-CTC inference"""
        # greedy ctc outputs
        enc_out = enc_out.unsqueeze(0)
        ctc_logit = torch.exp(self.ctc.log_softmax(enc_out))
        ctc_ids = ctc_logit.argmax(dim=-1)
        y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

        logging.info("ctc:{}".format(self.ids2text(y_hat[y_idx].tolist())))
        y_in = [y_hat[y_idx]]
        # calculate token-level ctc probabilities by taking
        # the maximum probability of consecutive frames with
        # the same ctc symbols
        ctc_hat = []
        #logit_prob
        cnt = 0
        for i, y in enumerate(y_hat.tolist()):
            ctc_hat.append(torch.tensor([-1]).cuda())
            while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                #breakpoint()
                if ctc_hat[i].max() < ctc_logit[0][cnt].max():
                    ctc_hat[i] = ctc_logit[0][cnt] #.item()
                    #logit_prob
                cnt += 1
        #breakpoint()
        ctc_hat = torch.stack(ctc_hat)[y_idx]
        ctc_hat[:, 0] = -1  #  blank id
        mask_idx = ctc_hat.max(-1)[0] < self.threshold_probability
        hypotheses = ctc_hat.topk(self.num_hypotheses,dim=-1)[1].transpose(0,1)
        y_hat = y_hat[y_idx].unsqueeze(0).repeat(self.num_hypotheses, 1)
        
        for i in range(1, self.num_hypotheses):
            if i == 1:  # masking
                y_hat[i][mask_idx] = self.mask_token
                
            else:  # top k replacement
                y_hat[i][mask_idx] = hypotheses[i-1][mask_idx]
            logging.info("hyp:{}".format(self.ids2text(y_hat[i].tolist())))
            y_in.append(y_hat[i])
        y_in = torch.stack(y_in, dim=0)
        for i in hypotheses[2:]:
            logging.info("hyp:{}".format(self.ids2text(i.tolist())))
        
        enc_length = torch.LongTensor([enc_out.size(1)])
        y_in_length = torch.LongTensor([y_in.size(1)])
        # iterative decoding
        # if not mask_num == 0:
        #     K = self.n_iterations
        #     num_iter = K if mask_num >= K and K > 0 else mask_num
        
        mask_idxs = mask_idx.unsqueeze(0).repeat(self.num_hypotheses,1)
        ctc_hat = torch.cat([ctc_hat,torch.zeros(ctc_hat.size(0),1).cuda()], -1)
        try:
            for t in range(self.n_iterations - 1):
                pred, _ = self.mlm(enc_out, enc_length,
                                    y_in, y_in_length)
                # cols = torch.arange(pred.size(1))
                
                # scores, ids = pred.max(-1)
                # max_score, max_ids = scores.max(0)
                #breakpoint()
                # new_pred = torch.stack([pred[max_ids, cols],
                #                         pred.median(0)[0],
                #                         pred.mean(0),
                #                         ])
                # pred = torch.cat((new_pred, pred), dim=0)
                topk_idx = pred.max(-1)[0].mean(1).topk(
                    min(self.num_hypotheses, pred.size(0)))[1]
                pred = pred[topk_idx]
                #breakpoint()
                pred = torch.softmax(pred,dim=-1) + 0.2 * ctc_hat
                #breakpoint()
                new_y_in = [] #y_in[topk_idx].unsqueeze(1).repeat(1,self.num_hypotheses,1).reshape(-1, y_in.size(1))
                new_mask_idxs =[]
                #breakpoint()
                for i in range(pred.size(0)):
                    for j in range(self.num_hypotheses):
                        if i ==0:
                            new_y_in.append(pred[i].argmax(dim=-1))
                        elif i == 1:
                            
                            _y = y_in[i].clone()
                            mask_idx = mask_idxs[i]
                            mask_num = mask_idx.sum().item()
                            k = mask_num//(self.n_iterations-t)
                            replace_idx = pred[i][mask_idx].max(-1)[0].topk(k, dim=-1)[1]
                            _y[mask_idx][replace_idx] = pred[i][mask_idx].argmax(dim=-1)[replace_idx]
                            new_y_in.append(_y)
                            mask_idx = _y == self.mask_token
                            new_mask_idxs.append(mask_idx)
                        else:  #
                            
                            _y = y_in[i].clone()
                            replace_idx = pred[i].max(-1)[0].topk(pred.size(1))[1][-int(y_in.size(1)*0.1)-1:]
                            _y[replace_idx] = pred[i][replace_idx].topk(j+1, dim=-1)[1][:,j]
                            new_y_in.append(_y)
                y_in = torch.stack(new_y_in, dim=0)
                mask_idxs = torch.stack(new_mask_idxs, dim=0).repeat(self.num_hypotheses,1)
                
                #breakpoint()
                # # y_in_length = y_in_length[topk_idx].unsqueeze(-1).repeat(1, 1, self.num_hypotheses).transpose(1, 2).reshape(
                # #         -1,  y_in.size(1))
                # breakpoint()
                # dist = torch.distributions.Categorical(logits=pred[topk_idx])
                # #breakpoint()
                # y_in = torch.cat([dist.sample() for _ in range(self.num_hypotheses)], dim=0)
                #mask_prob = torch.rand(y_in.shape)
                #y_in[mask_prob < 0.1] = self.mask_token
                logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

            # predict leftover masks (|masks| < mask_num // num_iter)
            pred, _ = self.mlm(enc_out, enc_length, y_in, y_in_length)
            if pred.size(0) > self.num_hypotheses:
                score, indices = pred.max(-1)[0].mean(1).topk(self.num_hypotheses)
                pred = pred[indices]
            y_in = pred.argmax(dim=-1)
        except:
            pass
        #  y_in[0][mask_idx] = pred[0][mask_idx].argmax(dim=-1)

        logging.info("res:{}".format(self.ids2text(y_in[0].tolist())))
            #breakpoint()
        # pad with mask tokens to ensure compatibility with sos/eos tokens
        yseq = torch.tensor(
            [self.mask_token] + y_in.tolist()[0] + [self.mask_token], device=y_in.device
        )

        return Hypothesis(yseq=yseq)


    def _foward_type3(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        enc_out = enc_out.unsqueeze(0)
        ctc_logit, ctc_ids = torch.exp(self.ctc.log_softmax(enc_out)).max(dim=-1)
        y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

        logging.info("ctc:{}".format(self.ids2text(y_hat[y_idx].tolist())))
        
        return Hypothesis(yseq=y_hat[y_idx])