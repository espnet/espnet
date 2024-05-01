import torch
import logging

from typing import Any, Dict, List, NamedTuple, Tuple, Union
from espnet2.speechlm.module.module import MultiHeadedAttention

class SpeechLMHypothesis(NamedTuple):
    """ Hypothesis data type. """

    prefix: torch.Tensor
    generated: torch.Tensor
    score: torch.Tensor

class SpeechLMInference(object):
    """ Object to implement SpeechLM inference """
    def __init__(
        self,
        corelm,
        predictor,
        emb,
        device,
        token_list,
        token_bias,
        search_type: str = "ar",
        search_algo: str = "sampling",
        nbest: int = 1,
        sampling_temperature: float = 1.0,
        top_k: int = 20,
        maxlenratio: float =10.0,
        minlenratio: float = 1.0,
        modality: str = "codec",
    ):
        # if teacher_force and nbest > 1:
        #     raise ValueError("Teacher force can only produce one result")
        
        if maxlenratio <= 0:
            raise ValueError("maxlenratio has to be positive")

        self.corelm = corelm
        self.predictor = predictor
        self.emb = emb
        self.device = device
        self.token_list = token_list.copy()
        self.token_bias = token_bias
        self.search_type = search_type
        self.search_algo = search_algo
        self.nbest = nbest
        self.sampling_temerature = sampling_temperature
        self.top_k = top_k
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.modality = modality

        self.eos = self.token_list.index("<sos/eos>")
        self.valid_start = self.token_bias[modality]
        self.valid_end = min(
            [s for s in self.token_bias.values() if s > self.valid_start] + [len(token_list)]
        )
    
    @torch.no_grad()
    def __call__(
        self, 
        dec_seq: torch.Tensor,
        prefix_len: int,
        enc_seq: torch.Tensor = None,
    ) -> List[SpeechLMHypothesis]:
        method_name = f"inference_{self.search_type}"
        return getattr(self, method_name)(dec_seq, prefix_len, enc_seq)

    def logits_to_token(
        self,
        logits: torch.Tensor,
        suffix: torch.Tensor,
        idx: int,
        minlen: int,
    ):
        # (1) In the valid range: target modality or <sos/eos>
        mask = torch.ones_like(logits).bool()

        if self.modality == "codec":
            nq = logits.size(2)
            assert (self.valid_start - self.valid_end) % nq  == 0
            increment = (self.valid_end - self.valid_start) // nq
            start = self.valid_start
            for layer_idx in range(nq):
                mask[:, :, layer_idx, start: start+increment] = False
                start += increment
        else:
            mask[:, :, :, self.valid_start: self.valid_end] = False

        if idx >= minlen:
            mask[:, :, :, self.eos] = False
        logits = logits.masked_fill_(mask, -1e20)

        # (2) top_k selection
        top_k = self.top_k if self.top_k > 0 else logits.size(-1)
        topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)

        if self.search_algo == "sampling":
            # (3) temperature
            logp = torch.softmax(
                topk_values / self.sampling_temerature,
                dim=-1
            ).flatten(end_dim=-2)
            inner_indices = torch.multinomial(logp, num_samples=1)
            gen_token_idx = torch.gather(topk_indices, inner_indices, dim=-1)
            gen_token_score = torch.gather(topk_values, inner_indices, dim=-1)
        
        elif self.search_algo in ["greedy_search", "teacher_force"]:
            gen_token_idx = topk_indices[:, :, :, 0]
            gen_token_score = topk_values[:, :, :, 0]
        
        else:
            raise NotImplementedError
        
        if self.search_algo == "teacher_force":
            if idx >= suffix.size(1):
                raise ValueError("Suffix is too short to do teacher force")
            prev_token = suffix[:, idx].unsqueeze(1)
        else:
            prev_token = gen_token_idx
        
        return gen_token_idx, gen_token_score, prev_token

    def inference_ar(
        self, 
        dec_seq: torch.Tensor,
        prefix_len: int,
        enc_seq: torch.Tensor = None,
    ) -> List[SpeechLMHypothesis]:
        
        # (1) Initialization
        self.corelm.init_cache()
        self.predictor.init_cache()

        # (2) Prefix inference
        dec_seq = dec_seq.expand(self.nbest, -1, -1)
        prefix_emb = self.emb(dec_seq[:, :prefix_len])
        _ = self.corelm(prefix_emb)
        if self.search_algo == "teacher_force":
            suffix = dec_seq[:, prefix_len + 1:] # skip modality start token
        else:
            suffix = None

        # (3) Start for loop
        minlen = int(prefix_len * self.minlenratio) if self.minlenratio > 0 else 0
        maxlen = int(prefix_len * self.maxlenratio)
        if self.search_algo == "teacher_force":
            minlen = len(suffix[0])
            maxlen = len(suffix[0])

        nq = dec_seq.size(2)
        start_tok = self.token_list.index(f"<{self.modality}_start/end>")
        prev_tok = torch.Tensor([start_tok]).expand(1, 1, nq).to(self.device).long()

        generated = {"token": [], "score": []}
        finish_idx = torch.Tensor([-1]).expand(self.nbest).long().to(self.device)
        for idx in range(maxlen):
            # (3.1) forward one step
            prev_tok_emb = self.emb(prev_tok)
            this_hidden, _, _ = self.corelm(prev_tok_emb)
            this_logits, _, _ = self.predictor(this_hidden)
            gen_tok, gen_score, prev_tok = self.logits_to_token(
                this_logits, suffix, idx, minlen,
            )
            generated['token'].append(gen_tok)
            generated['score'].append(gen_score)

            # (3.2) detect ended hypotheses.
            finish_idx = torch.where(
                torch.any(prev_tok == self.eos),
                idx,
                finish_idx,
            )

            if torch.all(torch.ge(finish_idx, 0)):
                logging.info(f"Finish generation with sample lengths: {finish_idx.cpu().tolist()}")
                break

        # (4) finalize
        self.corelm.remove_cache()
        self.predictor.remove_cache()

        # (5) organize hypotheses
        generated = {
            "token": torch.cat(generated["token"], dim=1),
            "score": torch.cat(generated["score"], dim=1)
        }
        hypos = [
            SpeechLMHypothesis(
                prefix=dec_seq[b, :prefix_len],
                generated=generated["token"][b][:finish_idx[b]],
                score=generated["score"][b][:finish_idx[b]]
            )
            for b in range(self.nbest)
        ]

        return hypos

    def inference_nar(
        self, 
        dec_seq: torch.Tensor,
        prefix_len: int,
        enc_seq: torch.Tensor = None,
    ) -> List[SpeechLMHypothesis]:
        pass

    def inference_arnar(
        self, 
        dec_seq: torch.Tensor,
        prefix_len: int,
        enc_seq: torch.Tensor = None,
    ) -> List[SpeechLMHypothesis]:
        pass
    
    def train_forward(
        self, 
        dec_seq: torch.Tensor,
        prefix_len: int,
        enc_seq: torch.Tensor = None,
    ) -> List[SpeechLMHypothesis]:
        from espnet2.speechlm.net_utils import length_mask
        target = dec_seq[:, 1:]
        target_emb = self.emb(target)

        dec_seq = dec_seq[:, :-1]
        dec_emb = self.emb(dec_seq)

        dec_length = torch.Tensor([dec_emb.size(1)]).long().to(self.device)

        pred, pred_lengths, others = self.corelm(dec_emb, dec_length)

        logits, logits_lengths, others = self.predictor(
            pred,
            pred_lengths,
            target_emb,
            pred_lengths,
            others,
        )

        target_sequence, target_sequence_lengths, others = self.predictor.organize_target(
            target, dec_length, others
        )

        elem_loss = torch.nn.functional.cross_entropy(
            logits.permute(0, 3, 1, 2), target_sequence, reduction="none"
        )

        mask = length_mask(logits_lengths).to(elem_loss.dtype).unsqueeze(-1)
        elem_loss = elem_loss * mask
        loss = elem_loss.sum() / mask.sum() / logits.size(2)

        pred = logits.argmax(dim=-1)
        acc = torch.eq(pred, target_sequence).to(elem_loss.dtype) * mask

        stats = {}
        for nq_idx in range(target_sequence.size(2)):
            stats.update({
                f"acc_layer{nq_idx}": acc[:, :, nq_idx].float().sum() / mask.float().sum()
            })
        
        acc = acc.sum() / mask.sum() / logits.size(2)
        stats.update({"loss": loss.clone().detach(), "acc": acc})


        
        print('training stats: ', stats)
