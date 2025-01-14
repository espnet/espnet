import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats
import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.aai.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetAAIModel(AbsESPnetModel):
    """Encoder model"""

    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        postencoder: Optional[AbsPostEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ignore_id: int = -1,
        length_normalized_loss: bool = False,
        extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()

        super().__init__()

        self.ignore_id = ignore_id
        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.postencoder = postencoder
        self.decoder = decoder
        self.error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__

        if self.is_encoder_whisper:
            assert (
                self.frontend is None
            ), "frontend should be None when using full Whisper model"

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        ema: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            ema: (Batch, Length, ...)
            kwargs: "utt_id" is among the input.
        """
        # Check that batch_size is unified
        assert speech.shape[0] == speech_lengths.shape[0] == ema.shape[0], (
            speech.shape,
            speech_lengths.shape,
            ema.shape,
        )
        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        encoder_out = self.decoder(encoder_out, encoder_out_lens)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_mse = None
        stats = dict()

        loss_mse, cc = self._calc_aai_loss(encoder_out, encoder_out_lens, ema)

        stats["loss_mse"] = loss_mse.detach()
        stats["cc"] = cc
        stats["loss"] = loss_mse.detach()

        loss, stats, weight = force_gatherable(
            (loss_mse, stats, batch_size), loss_mse.device
        )
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        ema: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # example - upsample features
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

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
        return encoder_out, encoder_out_lens

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

    def _calc_aai_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
    ):
        lossfn = torch.nn.MSELoss(reduction="mean")

        # maybe not a good idea to do this..
        lens = min(encoder_out.shape[1], ys_pad.shape[1])
        encoder_out = encoder_out[:, :lens, :]
        ys_pad = ys_pad[:, :lens, :]
        encoder_out_lens = [min(x, lens) for x in encoder_out_lens]

        if not self.training:
            cc = []
            ys_pad_ = ys_pad.detach().cpu().numpy()
            encoder_out_ = encoder_out.detach().cpu().numpy()
        else:
            cc = None
        # change to mask based normalisation instead of loop
        for i in range(encoder_out.shape[0]):
            if i == 0:
                loss = lossfn(
                    encoder_out[i, : encoder_out_lens[i], :],
                    ys_pad[i, : encoder_out_lens[i], :],
                )
            else:
                loss = loss + lossfn(
                    encoder_out[i, : encoder_out_lens[i], :],
                    ys_pad[i, : encoder_out_lens[i], :],
                )
            #
            if not self.training:
                c = []
                # change to torch cc
                for j in range(encoder_out.shape[-1]):
                    single_utt_cc = scipy.stats.pearsonr(
                        ys_pad_[i, : encoder_out_lens[i], j],
                        encoder_out_[i, : encoder_out_lens[i], j],
                    )[0]
                    c.append(single_utt_cc)
                cc.append(c)
        loss = loss / encoder_out.shape[0]
        if not self.training:
            cc = np.array(cc)
            cc = np.mean(np.mean(cc, axis=0))

        return loss, cc
