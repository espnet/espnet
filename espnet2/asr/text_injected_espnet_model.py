import random

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F

from packaging.version import parse as V

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


def mask_input(
    input_tensor: torch.Tensor,
    mask_prob: float=0.15,
    mask_span: int=5,
) -> torch.Tensor:
    """Masks out a percentage of the input_tensor with a specified span.
    Args:
        input_tensor (torch.Tensor): The input tensor to mask.
        mask_prob (float, optional): The probability of masking out a token. Defaults to 0.15.
        mask_span (int, optional): The span of the masked out region. Defaults to 5.
    Returns:
        torch.Tensor: The masked out tensor.
    """
    input_length = input_tensor.size(1)
    mask_indices = torch.zeros_like(input_tensor, dtype=torch.bool)
    for index in range(input_length):
        if random.random() < mask_prob:
            mask_indices[:, index: index + mask_span] = True
    masked_input = input_tensor.masked_fill(mask_indices, 0)

    return masked_input


class TextInjectedESPnetASRModel(ESPnetASRModel):
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
        injected_text_frequency: int = 3,
        injected_text_type: str = "fixed",
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

        self.injected_text_frequency = injected_text_frequency
        self.injected_text_type = injected_text_type
        self.injected_text_embedding = torch.nn.Embedding(vocab_size, frontend.n_mels)
        
        self.injected_statistics = None
        if self.injected_text_type in ["mean", "median", "normal", "mean_median"]:
            injected_statistics = np.load("./exp/asr_stats_raw_en_bpe5000_sp/train/token_statistics.npy")
            self.injected_statistics = torch.from_numpy(injected_statistics).cuda()

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
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

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        pseudo_mask = speech_lengths == 0

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(
            speech,
            speech_lengths,
            text,
            text_lengths,
        )

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None

        stats = dict()

        pseudo_loss_ctc, pseudo_cer_ctc, speech_loss_ctc, speech_cer_ctc = None, None, None, None
        pseudo_loss_att, pseudo_acc_att, pseudo_cer_att, pseudo_wer_att = None, None, None, None
        speech_loss_att, speech_acc_att, speech_cer_att, speech_wer_att = None, None, None, None

        pseudo_encoder_out, pseudo_encoder_out_lens = encoder_out[pseudo_mask], encoder_out_lens[pseudo_mask]
        speech_encoder_out, speech_encoder_out_lens = encoder_out[~pseudo_mask], encoder_out_lens[~pseudo_mask]

        speech_num = speech_encoder_out.shape[0] if speech_encoder_out is not None else 0
        pseudo_num = pseudo_encoder_out.shape[0] if pseudo_encoder_out is not None else 0
        pseudo_weight = 0.3

        stats["speech_num"] = speech_num
        stats["pseudo_num"] = pseudo_num
        stats["text_len"] = torch.mean(text_lengths.float())

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            if pseudo_encoder_out.shape[0] > 0:
                pseudo_loss_ctc, pseudo_cer_ctc = self._calc_ctc_loss(
                    pseudo_encoder_out,
                    pseudo_encoder_out_lens,
                    text[pseudo_mask],
                    text_lengths[pseudo_mask],
                )
                stats["pseudo_loss_ctc"] = pseudo_loss_ctc.detach() if pseudo_loss_ctc is not None else None
                stats["pseudo_cer_ctc"] = pseudo_cer_ctc
                stats["pseudo_speech_len"] = torch.mean(pseudo_encoder_out_lens.float())
                stats["pseudo_target_len"] = torch.mean(text_lengths[pseudo_mask].float())

            if speech_encoder_out.shape[0] > 0:
                speech_loss_ctc, speech_cer_ctc = self._calc_ctc_loss(
                    speech_encoder_out,
                    speech_encoder_out_lens,
                    text[~pseudo_mask],
                    text_lengths[~pseudo_mask],
                )
                stats["speech_loss_ctc"] = speech_loss_ctc if speech_loss_ctc is not None else None
                stats["speech_cer_ctc"] = speech_cer_ctc
                stats["speech_len"] = torch.mean(speech_encoder_out_lens.float())
                stats["speech_target_len"] = torch.mean(text_lengths[~pseudo_mask].float())

            pseudo_loss_ctc = pseudo_loss_ctc if pseudo_loss_ctc is not None else 0
            speech_loss_ctc = speech_loss_ctc if speech_loss_ctc is not None else 0
            loss_ctc = (pseudo_weight * pseudo_num * pseudo_loss_ctc + speech_num * speech_loss_ctc) / (pseudo_num + speech_num)
            # loss_ctc = pseudo_weight * pseudo_loss_ctc + speech_loss_ctc
            # loss_ctc, cer_ctc = self._calc_ctc_loss(
            #     encoder_out, encoder_out_lens, text, text_lengths
            # )

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
                # loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                #     encoder_out, encoder_out_lens, text, text_lengths
                # )

                if pseudo_encoder_out.shape[0] > 0:
                    (
                        pseudo_loss_att,
                        pseudo_acc_att,
                        pseudo_cer_att,
                        pseudo_wer_att,
                    ) = self._calc_att_loss(
                        pseudo_encoder_out,
                        pseudo_encoder_out_lens,
                        text[pseudo_mask],
                        text_lengths[pseudo_mask],
                    )

                    stats["pseudo_loss_att"] = pseudo_loss_att.detach() if pseudo_loss_att is not None else None
                    stats["pseudo_acc_att"] = pseudo_acc_att
                    stats["pseudo_cer_att"] = pseudo_cer_att
                    stats["pseudo_wer_att"] = pseudo_wer_att

                if speech_encoder_out.shape[0] > 0:
                    (
                        speech_loss_att,
                        speech_acc_att,
                        speech_cer_att,
                        speech_wer_att,
                    ) = self._calc_att_loss(
                        speech_encoder_out,
                        speech_encoder_out_lens,
                        text[~pseudo_mask],
                        text_lengths[~pseudo_mask],
                    )

                    stats["speech_loss_att"] = speech_loss_att.detach() if speech_loss_att is not None else None
                    stats["speech_acc_att"] = speech_acc_att
                    stats["speech_cer_att"] = speech_cer_att
                    stats["speech_wer_att"] = speech_wer_att

            speech_loss_att = speech_loss_att if speech_loss_att is not None else 0
            pseudo_loss_att = pseudo_loss_att if pseudo_loss_att is not None else 0
            # loss_att = pseudo_weight * pseudo_loss_att + speech_loss_att
            loss_att = (pseudo_weight * pseudo_num * pseudo_loss_att + speech_num * speech_loss_att) / (speech_num + pseudo_num)
            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss_ctc = loss_ctc if loss_ctc is not None else 0
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None

            speech_acc_att = speech_acc_att if speech_acc_att is not None else 0
            pseudo_acc_att = pseudo_acc_att if pseudo_acc_att is not None else 0
            stats["acc"] = (speech_num * speech_acc_att + pseudo_acc_att * pseudo_num) / (speech_num + pseudo_num)
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _extract_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        feats = speech[speech_lengths > 0]
        feats_lengths = speech_lengths[speech_lengths > 0]

        if feats.shape[0] > 0:
            # 1. Extract feats
            feats, feats_lengths = super()._extract_feats(feats, feats_lengths)
            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        if text is None:
            return feats, feats_lengths, None

        pseudo_feats = text[speech_lengths == 0]
        pseudo_feats[pseudo_feats == self.ignore_id] = 0

        pseudo_feats_lengths = text_lengths[speech_lengths == 0]
        pseudo_mask = torch.zeros((text.size(0)), dtype=torch.bool).to(text.device)

        if pseudo_feats.shape[0] > 0:
            if self.injected_text_type == "fixed":
                pseudo_feats = torch.repeat_interleave(
                    pseudo_feats,
                    self.injected_text_frequency,
                    dim=1,
                )
            elif self.injected_text_type == "random":
                random_text_frequency = torch.randint(
                    low=self.injected_text_frequency - 2,
                    high=self.injected_text_frequency + 3,
                    size=(pseudo_feats.size(1),),
                    dtype=pseudo_feats.dtype,
                    device=pseudo_feats.device,
                )

                pseudo_feats = torch.repeat_interleave(
                    pseudo_feats,
                    random_text_frequency,
                    dim=1,
                )
            elif self.injected_text_type in ["mean", "median", "normal", "mean_median"]:
                if self.injected_text_type == "mean":
                    repeated_text_frequency = self.injected_statistics[pseudo_feats][:, :, 0].type(pseudo_feats.dtype)
                elif self.injected_text_type == "median":
                    repeated_text_frequency = self.injected_statistics[pseudo_feats][:, :, 2].type(pseudo_feats.dtype)
                elif self.injected_text_type == "normal":
                    mean_text_frequency = self.injected_statistics[pseudo_feats][:, :, 0].type(torch.float)
                    std_text_frequency = self.injected_statistics[pseudo_feats][:, :, 1].type(torch.float)
                    repeated_text_frequency = torch.normal(mean_text_frequency, std_text_frequency).type(pseudo_feats.dtype)
                    repeated_text_frequency[repeated_text_frequency <= 0] = 1
                    repeated_text_frequency[pseudo_feats_lengths == 0] = 0
                elif self.injected_text_type == "mean_median":
                    mean_prob = torch.FloatTensor(1).uniform_(0, 1.0)

                    if mean_prob > 0.5:
                        repeated_text_frequency = self.injected_statistics[pseudo_feats][:, :, 0].type(pseudo_feats.dtype)
                    else:
                        repeated_text_frequency = self.injected_statistics[pseudo_feats][:, :, 2].type(pseudo_feats.dtype)   

                batch_size = pseudo_feats.size(0)
                repeated_pseudo_feats = []

                max = 0
                for index in range(batch_size):
                    repeated_feats = torch.repeat_interleave(
                        pseudo_feats[index],
                        repeated_text_frequency[index],
                        dim=0,
                    )
                    repeated_pseudo_feats.append(repeated_feats)
                    feats_len = repeated_feats[repeated_feats != 0].size(0)
                    max = feats_len if feats_len > max else max

                for index in range(batch_size):
                    feats_len = repeated_pseudo_feats[index].size(0)

                    if max < feats_len:
                        repeated_pseudo_feats[index] = repeated_pseudo_feats[index][:max]
                    elif max > feats_len:
                        diff = max - feats_len
                        repeated_pseudo_feats[index] = F.pad(
                            repeated_pseudo_feats[index],
                            (0, diff),
                            value=0,
                        )
                pseudo_feats = torch.stack(repeated_pseudo_feats, dim=0)

            else:
                raise NotImplementedError

            pseudo_feats_lengths = torch.count_nonzero(pseudo_feats, dim=-1)

            # mask out 15% token
            pseudo_feats = mask_input(input_tensor=pseudo_feats)
            pseudo_feats = self.injected_text_embedding(pseudo_feats)

            if feats.shape[0] > 0:
                speech_length = feats.shape[1]
                pseudo_speech_length = pseudo_feats.shape[1]

                if speech_length > pseudo_speech_length:
                    diff = speech_length - pseudo_speech_length
                    pseudo_feats = F.pad(
                        pseudo_feats.transpose(1, 2), (0, diff),
                        value=0,
                    )
                    pseudo_feats = pseudo_feats.transpose(1, 2)
                elif pseudo_speech_length > speech_length:
                    diff = pseudo_speech_length - speech_length
                    feats = F.pad(
                        feats.transpose(1, 2), (0, diff),
                        value=0,
                    )
                    feats = feats.transpose(1, 2)

                feats = torch.cat([feats, pseudo_feats], dim=0)
                feats_lengths = torch.cat(
                    [feats_lengths, pseudo_feats_lengths],
                    dim=0,
                )
                pseudo_mask[feats.size(0):] = True
            else:
                feats = pseudo_feats
                feats_lengths = pseudo_feats_lengths
                pseudo_mask[:] = True

        return feats, feats_lengths, pseudo_mask

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        with autocast(False):
            feats, feats_lengths, pseudo_mask = self._extract_feats(
                speech,
                speech_lengths,
                text,
                text_lengths,
            )

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats,
                feats_lengths,
                pseudo_mask=pseudo_mask,
            )

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens
