"""SOT Whisper model for ESPnet2.

Extends ESPnetASRModel with uppercase min-CE loss for
case-invariant SOT training on native OpenAI Whisper encoder/decoder.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos


class SOTWhisperModel(ESPnetASRModel):
    """SOT Whisper model with optional uppercase min-CE loss.

    Inherits everything from ESPnetASRModel (encode, forward, etc.).
    Overrides _calc_att_loss to support min-CE over case variants.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: Optional[AbsEncoder],
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: Optional[dict] = None,
        ctc_weight: float = 0.0,
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
        sym_sos: str = "<|startoftranscript|>",
        sym_eos: str = "<|endoftext|>",
        autocast_frontend: bool = False,
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        use_uppercase_loss: bool = True,
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
            aux_ctc=aux_ctc,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            transducer_multi_blank_durations=transducer_multi_blank_durations,
            transducer_multi_blank_sigma=transducer_multi_blank_sigma,
            sym_sos=sym_sos,
            sym_eos=sym_eos,
            autocast_frontend=autocast_frontend,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,
        )

        self.use_uppercase_loss = use_uppercase_loss
        if use_uppercase_loss:
            self.upper_cased_tokens = self._create_lower_uppercase_mapping(token_list)
            logging.info(
                f"SOTWhisperModel: uppercase min-CE enabled, "
                f"{len(self.upper_cased_tokens)} case pairs"
            )
        else:
            self.upper_cased_tokens = {}

    @staticmethod
    def _create_lower_uppercase_mapping(
        token_list: Union[List[str], Tuple[str, ...]],
    ) -> dict:
        """Build mapping from lowercase token indices to uppercase token indices."""
        upper_cased = {}
        vocab = {token: idx for idx, token in enumerate(token_list)}

        for token, index in vocab.items():
            if len(token) < 1:
                continue
            if token[0] == "\u0120" and len(token) > 1:
                lower_cased_token = (
                    token[0] + token[1].lower() + (token[2:] if len(token) > 2 else "")
                )
            else:
                lower_cased_token = token[0].lower() + token[1:]
            if lower_cased_token != token:
                lower_index = vocab.get(lower_cased_token, None)
                if lower_index is not None:
                    upper_cased[lower_index] = index

        return upper_cased

    def _make_uppercase_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Create uppercase variant of labels for min-CE loss."""
        upp_labels = labels.clone()
        for lower_idx, upper_idx in self.upper_cased_tokens.items():
            upp_labels[labels == lower_idx] = upper_idx
        return upp_labels

    def _per_sample_min_ce(self, logits, labels, upp_labels):
        """Min-CE over case variants, matching HF's flat-mean computation.

        Args:
            logits:     [N, L, V]
            labels:     [N, L]
            upp_labels: [N, L]

        Returns:
            scalar loss
        """
        loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_id, reduction="none"
        )
        N, L, V = logits.shape
        flat_logits = logits.reshape(-1, V)
        nll1 = loss_fct(flat_logits, labels.reshape(-1))
        nll2 = loss_fct(flat_logits, upp_labels.reshape(-1))
        nll = torch.min(nll1, nll2)
        return nll.mean()

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            ys_pad = torch.cat(
                [
                    self.lang_token_id.repeat(ys_pad.size(0), 1).to(ys_pad.device),
                    ys_pad,
                ],
                dim=1,
            )
            ys_pad_lens += 1

        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # Compute loss
        if self.use_uppercase_loss:
            upp_labels = self._make_uppercase_labels(ys_out_pad)
            loss_att = self._per_sample_min_ce(decoder_out, ys_out_pad, upp_labels)
        else:
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
