from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postdecoder.abs_postdecoder import AbsPostDecoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRModel(AbsESPnetModel):
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
        postdecoder: Optional[AbsPostDecoder],
        deliberationencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        decoder2: Optional[AbsDecoder],
        ctc: CTC,
        rnnt_decoder: None,
        transcript_token_list: Union[Tuple[str, ...], List[str]] = None,
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        two_pass: bool = False,
        pre_postencoder_norm: bool = False,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()
        if transcript_token_list is not None:
            self.transcript_token_list = transcript_token_list.copy()
        # print(self.transcript_token_list)
        self.two_pass = two_pass
        print(self.two_pass)
        self.pre_postencoder_norm = pre_postencoder_norm
        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.postdecoder = postdecoder
        self.encoder = encoder
        if self.postdecoder is not None:
            print(self.encoder._output_size)
            print(self.postdecoder.output_size_dim)
            if self.encoder._output_size != self.postdecoder.output_size_dim:
                self.uniform_linear = torch.nn.Linear(
                    self.encoder._output_size, self.postdecoder.output_size_dim
                )
        self.decoder2 = decoder2
        self.deliberationencoder = deliberationencoder
        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        self.rnnt_decoder = rnnt_decoder
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        transcript: torch.Tensor = None,
        transcript_lengths: torch.Tensor = None,
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

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        if self.two_pass:
            (
                audio_encoder_out,
                audio_encoder_out_lens,
                encoder_out,
                encoder_out_lens,
            ) = self.encode(speech, speech_lengths, transcript, transcript_lengths)
            # 2a. Attention-decoder branch
            if self.ctc_weight == 1.0:
                loss_att, acc_att, cer_att, wer_att = None, None, None, None
            else:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    audio_encoder_out, audio_encoder_out_lens, text, text_lengths
                )
                if self.decoder2 is None:
                    loss_att2, acc_att2, cer_att2, wer_att2 = self._calc_att_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )
                else:
                    print("goo")
                    loss_att2, acc_att2, cer_att2, wer_att2 = self._calc_att_loss(
                        encoder_out,
                        encoder_out_lens,
                        text,
                        text_lengths,
                        use_decoder2=True,
                    )
                if loss_att is not None:
                    loss_att = (loss_att + loss_att2) / 2
                if acc_att is not None:
                    acc_att = (acc_att + acc_att2) / 2
                if cer_att is not None:
                    cer_att = (cer_att + cer_att2) / 2
                if wer_att is not None:
                    wer_att = (wer_att + wer_att2) / 2

            # 2b. CTC branch
            if self.ctc_weight == 0.0:
                loss_ctc, cer_ctc = None, None
            else:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    audio_encoder_out, audio_encoder_out_lens, text, text_lengths
                )
                loss_ctc2, cer_ctc2 = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )
                if loss_ctc is not None:
                    loss_ctc = (loss_ctc + loss_ctc2) / 2
                if cer_ctc is not None:
                    cer_ctc = (cer_ctc + cer_ctc2) / 2
        else:
            encoder_out, encoder_out_lens = self.encode(
                speech, speech_lengths, transcript, transcript_lengths
            )

            # 2a. Attention-decoder branch
            if self.ctc_weight == 1.0:
                loss_att, acc_att, cer_att, wer_att = None, None, None, None
            else:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 2b. CTC branch
            if self.ctc_weight == 0.0:
                loss_ctc, cer_ctc = None, None
            else:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        transcript: torch.Tensor = None,
        transcript_lengths: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        transcript_pad: torch.Tensor = None,
        transcript_pad_lens: torch.Tensor = None,
        device="cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(
            feats,
            feats_lengths,
            return_pos=False,
            pre_postencoder_norm=self.pre_postencoder_norm,
        )

        # Post-encoder, e.g. NLU
        # print(encoder_out.shape)
        # print(encoder_out_lens.shape)
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        if self.postdecoder is not None:
            if self.encoder._output_size != self.postdecoder.output_size_dim:
                encoder_out = self.uniform_linear(encoder_out)
            # print(transcript_pad.shape)
            # print(self.transcript_token_list)
            # print(self.transcript_token_list)
            # print(transcript_pad)
            transcript_list = [
                " ".join([self.transcript_token_list[int(k)] for k in k1 if k != -1])
                for k1 in transcript_pad
            ]
            # print("ok1")
            # transcript_len_list=[len(k) for k in transcript_list]
            (
                transcript_input_id_features,
                transcript_input_mask_features,
                transcript_segment_ids_feature,
                transcript_position_ids_feature,
                input_id_length,
            ) = self.postdecoder.convert_examples_to_features(transcript_list, 128)
            # print("ok")
            # print(np.array(transcript_input_id_features).shape)
            # print(transcript_input_id_features)
            bert_encoder_out = self.postdecoder(
                torch.LongTensor(transcript_input_id_features).to(device=device),
                torch.LongTensor(transcript_input_mask_features).to(device=device),
                torch.LongTensor(transcript_segment_ids_feature).to(device=device),
                torch.LongTensor(transcript_position_ids_feature).to(device=device),
            )
            # print(bert_encoder_out.shape)
            # print(encoder_out.shape)
            bert_encoder_lens = torch.LongTensor(input_id_length).to(device=device)
            bert_encoder_out = bert_encoder_out[:, : torch.max(bert_encoder_lens)]
            # print(encoder_out_lens.shape)
            # print(bert_encoder_lens)
            final_encoder_out_lens = encoder_out_lens + bert_encoder_lens
            # print(final_encoder_out_lens)
            max_lens = torch.max(final_encoder_out_lens)
            # print(max_lens)
            encoder_new_out = torch.zeros(
                (encoder_out.shape[0], max_lens, encoder_out.shape[2])
            ).to(device=device)
            for k in range(len(encoder_out)):
                # print(encoder_out[k,:encoder_out_lens[k]].shape)

                encoder_new_out[k] = torch.cat(
                    (
                        encoder_out[k, : encoder_out_lens[k]],
                        bert_encoder_out[k, : bert_encoder_lens[k]],
                        torch.zeros(
                            (max_lens - final_encoder_out_lens[k], encoder_out.shape[2])
                        ).to(device=device),
                    ),
                    0,
                )
            # # encoder_new_out=encoder_new_out.requires_grad_(True)
            # print(encoder_new_out.shape)
            # bert_encoder_out=bert_encoder_out.view(bert_encoder_out.shape[0],1,bert_encoder_out.shape[1])
            # bert_encoder_out=bert_encoder_out.expand(bert_encoder_out.shape[0],encoder_out.shape[1],bert_encoder_out.shape[2])
            # print(bert_encoder_out.shape)
            # encoder_out=torch.cat((encoder_out,\
            #         bert_encoder_out),1)
            if self.deliberationencoder is not None:
                encoder_new_out, final_encoder_out_lens = self.deliberationencoder(
                    encoder_new_out, final_encoder_out_lens
                )
            if not (self.two_pass):
                encoder_out = encoder_new_out
                encoder_out_lens = final_encoder_out_lens
            # encoder_out_lens[:]=encoder_out_lens.max()+torch.max(bert_encoder_lens)
            # print(encoder_out_lens)
            # exit()
            # print(torch.argmax(decoder_out_prob, dim=2))
            # print([self.token_list[k] for k in ys_out_pad[0]])
            # exit()

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )
        if self.two_pass:
            return (
                encoder_out,
                encoder_out_lens,
                encoder_new_out,
                final_encoder_out_lens,
            )
        else:
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

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        use_decoder2=False,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        if use_decoder2:
            decoder_out, _ = self.decoder2(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )
        else:
            decoder_out, _ = self.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
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

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError
