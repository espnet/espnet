import logging
import math
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.layers.gnn import GCN
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.text.Butils import BiasProc
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy, to_device
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetTCPGenBiasingASRModel(ESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

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
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        transducer_multi_blank_durations: List = [],
        transducer_multi_blank_sigma: float = 0.05,
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        biasing: bool = False,
        biasingsche: int = 0,
        battndim: int = 0,
        deepbiasing: bool = False,
        biasingGNN: str = "",
        biasinglist: str = "",
        bmaxlen: int = 100,
        bdrop: float = 0.0,
        bpemodel: str = "",
    ):
        super().__init__(
            vocab_size,
            token_list,
            frontend,
            specaug,
            normalize,
            preencoder,
            encoder,
            postencoder,
            decoder,
            ctc,
            joint_network,
            aux_ctc,
            ctc_weight,
            interctc_weight,
            ignore_id,
            lsm_weight,
            length_normalized_loss,
            report_cer,
            report_wer,
            sym_space,
            sym_blank,
            transducer_multi_blank_durations,
            transducer_multi_blank_sigma,
            sym_sos,
            sym_eos,
            extract_feats_in_collect_stats,
            lang_token_id,
        )
        # biasing
        if biasinglist != "":
            self.bprocessor = BiasProc(
                biasinglist,
                maxlen=bmaxlen,
                bdrop=bdrop,
                bpemodel=bpemodel,
                charlist=token_list,
            )
        self.biasing = biasing
        self.biasingsche = biasingsche
        self.GNN = biasingGNN

        if self.use_transducer_decoder:
            # tcpgen biasing
            if self.biasing:
                from warp_rnnt import rnnt_loss

                self.criterion_transducer_logprob = rnnt_loss

                self.attndim = battndim
                self.Qproj_acoustic = torch.nn.Linear(
                    self.encoder.output_size(),
                    self.attndim,
                )
                self.Qproj_char = torch.nn.Linear(self.decoder.dunits, self.attndim)
                self.Kproj = torch.nn.Linear(self.decoder.dunits, self.attndim)
                self.ooKBemb = torch.nn.Embedding(1, self.decoder.dunits)
                self.pointer_gate = torch.nn.Linear(
                    self.attndim + self.joint_network.joint_space_size,
                    1,
                )
                self.Bdrop = torch.nn.Dropout(0.1)
                self.deepbiasing = deepbiasing
                if self.GNN.startswith("gcn"):
                    self.gnn = GCN(
                        self.decoder.dunits,
                        self.decoder.dunits,
                        int(self.GNN[3:]),
                        0.1,
                        residual=True,
                        tied=True,
                    )
            else:
                from warprnnt_pytorch import RNNTLoss

                self.criterion_transducer = RNNTLoss(
                    blank=self.blank_id,
                    fastemit_lambda=0.0,
                )
        self.epoch = 0

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
            kwargs: "utt_id" is among the input.
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

        lextrees = []
        if self.biasing and self.epoch >= self.biasingsche:
            biasingwords, lextree = self.bprocessor.select_biasing_words(text.tolist())
            lextrees = [lextree] * text.size(0)

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out

                # use auxillary ctc data if specified
                loss_ic = None
                if self.aux_ctc is not None:
                    idx_key = str(layer_idx)
                    if idx_key in self.aux_ctc:
                        aux_data_key = self.aux_ctc[idx_key]
                        aux_data_tensor = kwargs.get(aux_data_key, None)
                        aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                        if aux_data_tensor is not None and aux_data_lengths is not None:
                            loss_ic, cer_ic = self._calc_ctc_loss(
                                intermediate_out,
                                encoder_out_lens,
                                aux_data_tensor,
                                aux_data_lengths,
                            )
                        else:
                            raise Exception(
                                "Aux. CTC tasks were specified but no data was found"
                            )
                if loss_ic is None:
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

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
                lextrees=lextrees,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
        lextrees: list = [],
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        # biasing
        trees = lextrees
        p_gen_mask_all = []
        KBembedding = []
        ptr_dist = []
        node_encs = None
        if self.biasing and self.epoch >= self.biasingsche:
            # Encode prefix tree using GNN
            if self.GNN != "":
                node_encs = self.gnn(lextrees[0], self.decoder.embed)

            # Forward TCPGen
            query_acoustic = self.Qproj_acoustic(encoder_out)
            for i in range(decoder_in.size(1)):
                retval = self.get_step_biasing_embs(
                    decoder_in[:, i], trees, lextrees, node_encs=node_encs
                )
                step_mask = retval[0]
                step_embs = retval[1]
                trees = retval[2]
                p_gen_mask = retval[3]
                back_transform = retval[4]
                index_list = retval[5]

                p_gen_mask_all.append(p_gen_mask)
                query_char = self.decoder.dropout_embed(
                    self.decoder.embed(decoder_in[:, i])
                )
                query_char = self.Qproj_char(query_char).unsqueeze(1)
                query = query_char + query_acoustic  # nutts * T * attn_dim
                hptr_i, tcpgen_dist_i = self.get_meetingKB_emb_map(
                    query, step_mask, back_transform, index_list, meeting_KB=step_embs
                )
                ptr_dist.append(tcpgen_dist_i.unsqueeze(2))
                KBembedding.append(hptr_i.unsqueeze(2))
            KBembedding = torch.cat(KBembedding, dim=2)
            ptr_dist = torch.cat(ptr_dist, dim=2)

        if self.biasing:
            joint_out, joint_acts = self.joint_network(
                encoder_out.unsqueeze(2),
                decoder_out.unsqueeze(1),
                hptr=KBembedding if self.epoch >= self.biasingsche else None,
            )
        else:
            joint_out = self.joint_network(
                encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
            )

        # biasing
        if self.biasing and self.epoch >= self.biasingsche:
            p_gen = torch.sigmoid(
                self.pointer_gate(torch.cat((joint_acts, KBembedding), dim=-1))
            )
            ptr_mask = to_device(self, torch.tensor(p_gen_mask_all)).t()
            p_gen = p_gen.masked_fill(ptr_mask.unsqueeze(1).unsqueeze(-1).bool(), 0)
            # Get factorised loss
            model_dist = torch.softmax(joint_out, dim=-1)
            p_not_null = 1.0 - model_dist[:, :, :, 0:1]
            ptr_dist_fact = ptr_dist[:, :, :, 1:] * p_not_null
            ptr_gen_complement = (ptr_dist[:, :, :, -1:]) * p_gen
            p_partial = ptr_dist_fact[:, :, :, :-1] * p_gen + model_dist[
                :, :, :, 1:
            ] * (1 - p_gen + ptr_gen_complement)
            p_final = torch.cat([model_dist[:, :, :, 0:1], p_partial], dim=-1)
            joint_out = torch.log(p_final + 1e-12)

        if self.biasing:
            joint_out = torch.log_softmax(joint_out, dim=-1)
            loss_transducer = self.criterion_transducer_logprob(
                joint_out,
                target,
                t_len,
                u_len,
                reduction="mean",
                blank=self.blank_id,
                gather=True,
            )
        else:
            loss_transducer = self.criterion_transducer(
                joint_out,
                target,
                t_len,
                u_len,
            )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer

    def _calc_batch_ctc_loss(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        if self.ctc is None:
            return
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # Calc CTC loss
        do_reduce = self.ctc.reduce
        self.ctc.reduce = False
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        self.ctc.reduce = do_reduce
        return loss_ctc

    def get_step_biasing_embs(self, char_ids, trees, origTries, node_encs=None):
        ooKB_id = self.vocab_size
        p_gen_mask = []
        maxlen = 0
        index_list = []
        new_trees = []
        masks_list = []
        nodes_list = []
        step_embs = []
        for i, vy in enumerate(char_ids):
            new_tree = trees[i][0]
            vy = vy.item()
            if vy == self.blank_id or vy == self.eos:
                new_tree = origTries[i]
                p_gen_mask.append(0)
            elif self.token_list[vy].endswith("▁"):
                if vy in new_tree and new_tree[vy][0] != {}:
                    new_tree = new_tree[vy]
                else:
                    new_tree = origTries[i]
                p_gen_mask.append(0)
            elif vy not in new_tree:
                new_tree = [{}]
                p_gen_mask.append(1)
            else:
                new_tree = new_tree[vy]
                p_gen_mask.append(0)
            new_trees.append(new_tree)
            if len(new_tree[0].keys()) > maxlen:
                maxlen = len(new_tree[0].keys())
            index_list.append(list(new_tree[0].keys()))
            if node_encs is not None:
                nodes_list.append([value[4] for key, value in new_tree[0].items()])

        maxlen += 1
        step_mask = []
        back_transform = torch.zeros(
            len(new_trees), maxlen, ooKB_id + 1, device=char_ids.device
        )
        ones_mat = torch.ones(back_transform.size(), device=char_ids.device)
        for i, indices in enumerate(index_list):
            step_mask.append(
                len(indices) * [0] + (maxlen - len(indices) - 1) * [1] + [0]
            )
            if node_encs is not None:
                nodes_list[i] = nodes_list[i] + [node_encs.size(0)] * (
                    maxlen - len(indices)
                )
            indices += [ooKB_id] * (maxlen - len(indices))
        step_mask = torch.tensor(step_mask).byte().to(char_ids.device)
        index_list = torch.LongTensor(index_list).to(char_ids.device)
        back_transform.scatter_(dim=-1, index=index_list.unsqueeze(-1), src=ones_mat)
        if node_encs is not None:
            node_encs = torch.cat([node_encs, self.ooKBemb.weight], dim=0)
            step_embs = node_encs[torch.tensor(nodes_list).to(node_encs.device)]

        return step_mask, step_embs, new_trees, p_gen_mask, back_transform, index_list

    def get_meetingKB_emb_map(
        self,
        query,
        meeting_mask,
        back_transform,
        index_list,
        meeting_KB=[],
    ):
        if meeting_KB == []:
            meeting_KB = torch.cat(
                [self.decoder.embed.weight.data, self.ooKBemb.weight], dim=0
            )
            meeting_KB = meeting_KB[index_list]
        meeting_KB = self.Bdrop(self.Kproj(meeting_KB))
        KBweight = torch.einsum("ijk,itk->itj", meeting_KB, query)
        KBweight = KBweight / math.sqrt(query.size(-1))
        KBweight.masked_fill_(
            meeting_mask.bool().unsqueeze(1).repeat(1, query.size(1), 1), -1e9
        )
        KBweight = torch.nn.functional.softmax(KBweight, dim=-1)
        if meeting_KB.size(1) > 1:
            KBembedding = torch.einsum(
                "ijk,itj->itk", meeting_KB[:, :-1, :], KBweight[:, :, :-1]
            )
        else:
            KBembedding = KBweight.new_zeros(
                meeting_KB.size(0), query.size(1), meeting_KB.size(-1)
            )
        KBweight = torch.einsum("ijk,itj->itk", back_transform, KBweight)
        return KBembedding, KBweight
