import logging
from contextlib import contextmanager
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

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


class AlignRefineModel(ESPnetASRModel):
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
        joint_network: Optional[torch.nn.Module] = None,
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        align_refine_k=4,
    ):
        assert check_argument_types()
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
            joint_network=None,
            ctc_weight=ctc_weight,
            interctc_weight=0.0,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
        )

        self.ctc = ctc  # self.ctc might be None if ctc_weight is 0

        self.align_refine_k = align_refine_k
        assert self.align_refine_k >= 1
        self.align_refine_ctc = CTC(
            odim=self.vocab_size,
            # assuming decoder has the same output size as encoder
            encoder_output_size=self.encoder.output_size(),
            dropout_rate=0.1,
        )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Align refine forward pass

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

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_ctc, cer_ctc = None, None
        loss_align_refine = None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2. Align refine
        if self.ctc_weight != 1.0:
            ys_hat = self.ctc.argmax(encoder_out)

            # Run k iterations and average the loss
            loss_align_refine = 0.0
            for i in range(self.align_refine_k):
                decoder_out, ys_hat = self.run_align_refine_once(
                    encoder_out, encoder_out_lens, ys_hat, encoder_out_lens
                )
                loss_iter, cer_iter = self._calc_align_refine_loss(
                    decoder_out,
                    encoder_out_lens,
                    ys_hat,
                    text,
                    text_lengths,
                )
                loss_align_refine = loss_align_refine + loss_iter

                stats[f"loss_ar_{i}"] = loss_iter.detach()
                stats[f"cer_ar_{i}"] = cer_iter

            loss_align_refine = loss_align_refine / self.align_refine_k

        # 3. CTC-AR loss definition
        if self.ctc_weight == 0.0:
            loss = loss_align_refine
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = (
                self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_align_refine
            )

        # Collect stats
        if loss_ctc is not None:
            stats["loss_ctc"] = loss_ctc.detach()
        if cer_ctc is not None:
            stats["cer_ctc"] = cer_ctc
        if loss_align_refine is not None:
            stats["loss_align_refine"] = loss_align_refine.detach()

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_align_refine_loss(
        self,
        decoder_out: torch.Tensor,
        decoder_out_lens: torch.Tensor,
        ys_hat: torch.Tensor,
        text: torch.Tensor,
        text_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        loss_ctc = self.align_refine_ctc(decoder_out, decoder_out_lens, text, text_lens)

        cer = None
        if not self.training and self.error_calculator is not None:
            ys_hat_ = ys_hat.detach().cpu()
            text_ = text.cpu()
            cer = self.error_calculator(ys_hat_, text_, is_ctc=True)

        return loss_ctc, cer

    def run_align_refine_once(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_hat: torch.Tensor,
        ys_hat_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        decoder_out, _ = self.decoder(
            encoder_out,
            encoder_out_lens,
            ys_hat,
            ys_hat_lens,
        )
        ys_hat = self.align_refine_ctc.argmax(decoder_out)
        return decoder_out, ys_hat
