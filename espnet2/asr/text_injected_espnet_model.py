import random
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
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
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


def mask_input(
    input_tensor: torch.Tensor,
    mask_prob: float = 0.15,
    mask_span: int = 5,
    mask_id: int = 0,
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
            mask_indices[:, index : index + mask_span] = True
    masked_input = input_tensor.masked_fill(mask_indices, mask_id)

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
        injected_vocab_sizes: List[int] = [],
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

        self.injected_statistics = None
        if self.injected_text_type in ["mean", "median", "normal", "mean_median"]:
            injected_statistics = np.load(
                "./exp/asr_stats_raw_en_bpe5000_sp/train/token_statistics.npy"
            )
            self.injected_statistics = torch.from_numpy(injected_statistics).cuda()

        self.injected_text_embedding = self.decoder.embed

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: Optional[torch.Tensor],
        text_lengths: Optional[torch.Tensor],
        injected_text: torch.Tensor,
        injected_text_lengths: torch.Tensor,
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
            == injected_text.shape[0]
            == injected_text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        batch_size = speech.shape[0]

        injected_text_type_idx = kwargs.get("injected_text_type_idx", None)
        injected_text_mask = kwargs.get("injected_text_mask", None)

        # for data-parallel
        text[text == -1] = self.ignore_id
        text = text[:, : text_lengths.max()]

        injected_text[injected_text == -1] = self.ignore_id
        injected_text = injected_text[:, : injected_text_lengths.max()]

        paired_num = torch.count_nonzero(text_lengths)
        text_injected_num = torch.count_nonzero(injected_text_lengths)
        speech_injected_num = 0

        # 1. Encoder
        if text_injected_num > 0:
            (
                text_injected_encoder_out,
                text_injected_encoder_out_lens,
            ) = self.text_injected_encode(
                injected_text,
                injected_text_lengths,
            )

        if paired_num > 0:
            (
                paired_encoder_out,
                paired_encoder_out_lens,
            ) = self.encode(
                speech,
                speech_lengths,
            )

        text_injected_loss_ctc = None
        speech_injected_loss_ctc = None
        paired_loss_ctc = None

        loss_ctc = None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            if text_injected_num > 0:
                text_injected_loss_ctc, text_injected_cer_ctc = self._calc_ctc_loss(
                    text_injected_encoder_out,
                    text_injected_encoder_out_lens,
                    injected_text,
                    injected_text_lengths,
                )

            if paired_num > 0:
                paired_loss_ctc, paired_cer_ctc = self._calc_ctc_loss(
                    paired_encoder_out,
                    paired_encoder_out_lens,
                    text,
                    text_lengths,
                )

        # TODO: Intermediate CTC (optional)

        text_injected_loss_att, speech_injected_loss_att, paired_loss_att = (
            None,
            None,
            None,
        )
        text_injected_acc_att, speech_injected_acc_att, paired_acc_att = (
            None,
            None,
            None,
        )

        # TODO: transducer decoder branch

        # 2a. Attention decoder branch
        if self.ctc_weight != 1.0:
            if text_injected_num > 0:
                (
                    text_injected_loss_att,
                    text_injected_acc_att,
                    text_injected_cer_att,
                    text_injected_wer_att,
                ) = self._calc_att_loss(
                    text_injected_encoder_out,
                    text_injected_encoder_out_lens,
                    injected_text,
                    injected_text_lengths,
                )

            if paired_num > 0:
                (
                    paired_loss_att,
                    paired_acc_att,
                    paired_cer_att,
                    paired_wer_att,
                ) = self._calc_att_loss(
                    paired_encoder_out,
                    paired_encoder_out_lens,
                    text,
                    text_lengths,
                )

        # 2.5 Calculate L_paired and L_unpaired
        loss_ctc = (
            paired_loss_ctc
            if paired_loss_ctc is not None
            else 0
            + 0.3
            * (
                speech_injected_loss_ctc
                if speech_injected_loss_ctc is not None
                else 0 + text_injected_loss_ctc
                if text_injected_loss_ctc is not None
                else 0
            )
        )
        loss_att = (
            paired_loss_att
            if paired_loss_att is not None
            else 0
            + 0.3
            * (
                speech_injected_loss_att
                if speech_injected_loss_att is not None
                else 0 + text_injected_loss_att
                if text_injected_loss_att is not None
                else 0
            )
        )
        acc_att = (
            (paired_acc_att if paired_acc_att is not None else 0) * paired_num
            + (speech_injected_acc_att if speech_injected_acc_att is not None else 0)
            * speech_injected_num
            + (text_injected_acc_att if text_injected_acc_att is not None else 0)
            * text_injected_num
        ) / batch_size

        # 3. Loss computation
        if self.ctc_weight > 0.0:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        else:
            loss = loss_att

        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
        stats["acc"] = acc_att

        stats["text_injected_num"] = text_injected_num.detach()
        # stats["speech_injected_num"] = speech_injected_num
        stats["paired_num"] = paired_num.detach()

        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return loss, stats, weight

    def text_injected_encode(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._text_injected_extract_feats(text, text_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        # encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats,
                feats_lengths,
                ctc=self.ctc,
                is_text_injected=True,
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats,
                feats_lengths,
                is_text_injected=True,
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

        assert encoder_out.size(0) == feats.size(0), (
            encoder_out.size(),
            feats.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (
                (encoder_out, intermediate_outs),
                encoder_out_lens,
                text,
                text_lengths,
            )

        return encoder_out, encoder_out_lens

    def _text_injected_extract_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # repetition
        if self.injected_text_type in ["fixed", "fixed_or_speech"]:
            pseudo_feats = torch.repeat_interleave(
                text,
                self.injected_text_frequency,
                dim=1,
            )
            pseudo_feats_lengths = text_lengths * self.injected_text_frequency
        elif self.injected_text_type in ["mean", "median", "normal"]:
            if self.injected_text_type == "mean":
                repeated_text_frequency = self.injected_statistics[text][:, :, 0].type(
                    text.dtype
                )
            elif self.injected_text_type == "median":
                repeated_text_frequency = self.injected_statistics[text][:, :, 2].type(
                    text.dtype
                )
            elif self.injected_text_type == "normal":
                mean_text_frequency = self.injected_statistics[text][:, :, 0].type(
                    torch.float
                )
                std_text_frequency = self.injected_statistics[text][:, :, 1].type(
                    torch.float
                )

                repeated_text_frequency = torch.normal(
                    mean_text_frequency, std_text_frequency
                ).type(text.dtype)
                repeated_text_frequency[repeated_text_frequency <= 0] = 0

            batch_size = text.size(0)
            repeated_pseudo_feats = []
            repeated_pseudo_feats_lengths = []

            max = 0
            for index in range(batch_size):
                repeated_feats = torch.repeat_interleave(
                    text[index],
                    repeated_text_frequency[index],
                    dim=0,
                )
                repeated_pseudo_feats.append(repeated_feats)
                feats_len = repeated_feats[repeated_feats != self.ignore_id].size(0)
                repeated_pseudo_feats_lengths.append(feats_len)
                max = feats_len if feats_len > max else max

            for index in range(batch_size):
                feats_len = repeated_pseudo_feats[index].size(0)

                if max > feats_len:
                    diff = max - feats_len
                    repeated_pseudo_feats[index] = F.pad(
                        repeated_pseudo_feats[index],
                        (0, diff),
                        value=0,
                    )
            pseudo_feats = torch.stack(repeated_pseudo_feats, dim=0)
            pseudo_feats_lengths = torch.stack(pseudo_feats_lengths, dim=0)

            del repeated_pseudo_feats

        # mask out 15% tokens
        pseudo_feats = mask_input(pseudo_feats)

        if self.frontend is not None:
            pseudo_feats, _ = add_sos_eos(
                pseudo_feats, self.sos, self.eos, self.ignore_id
            )
            pseudo_feats_lengths = pseudo_feats_lengths + 1
        pseudo_feats = self.injected_text_embedding(pseudo_feats)

        return pseudo_feats, pseudo_feats_lengths
