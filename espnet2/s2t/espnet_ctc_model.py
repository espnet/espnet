from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.cuda.amp import autocast
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class ESPnetS2TCTCModel(AbsESPnetModel):
    """OWSM-CTC model"""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        prompt_encoder: AbsEncoder,
        ctc: CTC,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        sym_sos: str = "<sos>",
        sym_eos: str = "<eos>",
        sym_sop: str = "<sop>",  # start of prev
        sym_na: str = "<na>",  # not available
        extract_feats_in_collect_stats: bool = True,
        ctc_asr_only: List[bool] = [False],
    ):
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()

        self.blank_id = token_list.index(sym_blank)
        self.sos = token_list.index(sym_sos)
        self.eos = token_list.index(sym_eos)
        self.sop = token_list.index(sym_sop)
        self.na = token_list.index(sym_na)
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()
        self.ctc_asr_only = ctc_asr_only  # type of interctc

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.prompt_encoder = prompt_encoder

        self.embed = torch.nn.Embedding(vocab_size, self.prompt_encoder.output_size())
        self.pos_enc = PositionalEncoding(self.prompt_encoder.output_size(), 0.0)

        if self.encoder.output_size() != self.prompt_encoder.output_size():
            # used in encoder to inject task and lang tokens
            self.embed_proj = torch.nn.Linear(
                self.prompt_encoder.output_size(), self.encoder.output_size()
            )
            # applied to the output of prompt encoder
            self.prompt_proj = torch.nn.Linear(
                self.prompt_encoder.output_size(), self.encoder.output_size()
            )
        else:
            self.embed_proj = torch.nn.Identity()
            self.prompt_proj = torch.nn.Identity()

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.error_calculator = None

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )

        self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__

        if self.is_encoder_whisper:
            assert (
                self.frontend is None
            ), "frontend should be None when using full Whisper model"

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text_prev: torch.Tensor,
        text_prev_lengths: torch.Tensor,
        prefix: torch.Tensor,
        prefix_lengths: torch.Tensor,
    ):
        """Encode input speech."""

        # Forward prompt encoder
        text_prev[text_prev == -1] = self.eos
        memory, memory_lengths, _ = self.prompt_encoder(
            self.pos_enc(self.embed(text_prev)), text_prev_lengths
        )
        memory_mask = (~make_pad_mask(memory_lengths)[:, None, :]).to(memory.device)

        # Extract speech features
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Forward encoder
        encoder_out, encoder_out_lens, _ = self.encoder(
            feats,
            feats_lengths,
            ctc=self.ctc,
            prefix_embeds=self.embed_proj(self.embed(prefix)),
            memory=self.prompt_proj(memory),
            memory_mask=memory_mask,
        )
        return encoder_out, encoder_out_lens

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_prev: torch.Tensor,
        text_prev_lengths: torch.Tensor,
        text_ctc: torch.Tensor,
        text_ctc_lengths: torch.Tensor,
        prefix: torch.Tensor,
        prefix_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            text_prev: (Batch, Length)
            text_prev_lengths: (Batch,)
            text_ctc: (Batch, Length)
            text_ctc_lengths: (Batch,)
            prefix: (Batch, Length=2), <lang> and <task>
            prefix_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
            == text_prev.shape[0]
            == text_prev_lengths.shape[0]
            == text_ctc.shape[0]
            == text_ctc_lengths.shape[0]
            == prefix.shape[0]
            == prefix_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
            text_prev.shape,
            text_prev_lengths.shape,
            text_ctc.shape,
            text_ctc_lengths.shape,
            prefix.shape,
            prefix_lengths.shape,
        )
        batch_size = speech.shape[0]

        # -1 is used as padding index in collate fn
        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        encoder_out, encoder_out_lens = self.encode(
            speech, speech_lengths, text_prev, text_prev_lengths, prefix, prefix_lengths
        )

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            encoder_out, intermediate_outs = encoder_out

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

        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_asr_only[-1]:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text_ctc, text_ctc_lengths
            )
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # Collect CTC branch stats
        stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
        stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            assert len(self.ctc_asr_only) == len(intermediate_outs) + 1
            for (layer_idx, intermediate_out), asr_only in zip(
                intermediate_outs, self.ctc_asr_only
            ):
                if asr_only:
                    loss_ic, cer_ic = self._calc_ctc_loss(
                        intermediate_out, encoder_out_lens, text_ctc, text_ctc_lengths
                    )
                else:
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

        loss = loss_ctc

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_prev: torch.Tensor,
        text_prev_lengths: torch.Tensor,
        text_ctc: torch.Tensor,
        text_ctc_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

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

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc
