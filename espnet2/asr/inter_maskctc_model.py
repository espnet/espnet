from typing import Dict, List, Optional, Tuple, Union

import torch
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
from espnet2.torch_utils.device_funcs import force_gatherable

autocast_type = torch.float16
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    autocast_type = torch.bfloat16


class InterMaskCTCModel(ESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

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
        transducer_multi_blank_durations: List = [],
        transducer_multi_blank_sigma: float = 0.05,
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        autocast_frontend: bool = False,
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        mask_threshold: float = 0.99,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

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
        token_list = list(token_list)
        # NOTE (Shih-Lun): else case is for OpenAI Whisper ASR model,
        #                  which doesn't use <blank> token
        if sym_blank not in token_list:
            token_list.append(sym_blank)
        if sym_mask not in token_list:
            token_list.append(sym_mask)

        self.vocab_size = len(token_list)
        self.mask_token = token_list.index(sym_mask)

        self.token_list = token_list.copy()
        self.mask_threshold = mask_threshold

    @staticmethod
    def _add_optional_loss(total, value):
        if value is None:
            return total
        if total is None:
            return value
        return total + value

    def _calc_single_pred_loss_and_cer(
        self,
        pred: Dict[str, torch.Tensor],
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad_cpu: Optional[torch.Tensor] = None,
    ):
        loss_ctc = None
        loss_mlm = None
        cer_ctc = None
        cer_mlm = None

        if "ctc" in pred:
            ctc_logits = pred["ctc"]
            loss_ctc = self.ctc.loss_fn(ctc_logits, ys_pad, encoder_out_lens, ys_pad_lens)
            if ys_pad_cpu is not None and self.error_calculator is not None:
                cer_ctc = self.error_calculator(
                    ctc_logits.argmax(-1).cpu(), ys_pad_cpu, is_ctc=True
                )

        if "mlm" in pred:
            mlm_logits = pred["mlm"]
            loss_mlm = self.ctc.loss_fn(mlm_logits, ys_pad, encoder_out_lens, ys_pad_lens)
            if ys_pad_cpu is not None and self.error_calculator is not None:
                cer_mlm = self.error_calculator(
                    mlm_logits.argmax(-1).cpu(), ys_pad_cpu, is_ctc=True
                )

        return loss_ctc, loss_mlm, cer_ctc, cer_mlm


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

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        acc_att, cer_att, wer_att = None, None, None
        stats: Dict[str, torch.Tensor] = {}

        # 1. CTC branch
        if self.ctc is None:
            raise RuntimeError("InterMaskCTCModel requires CTC module")

        if self.ctc_weight != 0.0:
            loss_ctc, loss_mlm, cer_ctc, cer_mlm = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
        else:
            loss_ctc, loss_mlm, cer_ctc, cer_mlm = [], [], [], []

        # Intermediate CTC (optional)
        loss_interctc = None
        loss_intermlm = None
        interctc_count = 0
        if self.interctc_weight != 0.0 and len(self.intermediate_layer_idx) > 0:
            for idx, layer_idx in enumerate(self.intermediate_layer_idx):
                # Use auxiliary CTC data if specified for this intermediate layer.
                loss_ic = None
                loss_im = None
                cer_ic = None
                cer_im = None
                if self.aux_ctc is not None:
                    idx_key = str(layer_idx)
                    if idx_key in self.aux_ctc:
                        aux_data_key = self.aux_ctc[idx_key]
                        aux_data_tensor = kwargs.get(aux_data_key, None)
                        aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                        if aux_data_tensor is not None and aux_data_lengths is not None:
                            ys_pad_cpu = (
                                aux_data_tensor.cpu()
                                if (not self.training and self.error_calculator is not None)
                                else None
                            )
                            loss_ic, loss_im, cer_ic, cer_im = (
                                self._calc_single_pred_loss_and_cer(
                                self.ctc.intermediate_outs[idx],
                                aux_data_tensor,
                                aux_data_lengths,
                                encoder_out_lens,
                                ys_pad_cpu=ys_pad_cpu,
                            )
                            )
                        else:
                            raise Exception(
                                "Aux. CTC tasks were specified but no data was found"
                            )
                if loss_ic is None:
                    loss_ic, loss_im, cer_ic, cer_im = (
                        loss_ctc[idx],
                        loss_mlm[idx],
                        cer_ctc[idx],
                        cer_mlm[idx],
                    )

                loss_interctc = self._add_optional_loss(loss_interctc, loss_ic)
                loss_intermlm = self._add_optional_loss(loss_intermlm, loss_im)
                interctc_count += 1
                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic
                stats["loss_intermlm_layer{}".format(layer_idx)] = (
                    loss_im.detach() if loss_im is not None else None
                )
                stats["cer_intermlm_layer{}".format(layer_idx)] = cer_im
            if interctc_count > 0:
                if loss_interctc is not None:
                    loss_interctc = loss_interctc / interctc_count
                if loss_intermlm is not None:
                    loss_intermlm = loss_intermlm / interctc_count

        final_loss_ctc = loss_ctc[-1] if loss_ctc else None
        final_loss_mlm = loss_mlm[-1] if loss_mlm else None

        if loss_interctc is not None and final_loss_ctc is not None:
            final_loss_ctc = (
                (1 - self.interctc_weight) * final_loss_ctc
                + self.interctc_weight * loss_interctc
            )
        if loss_intermlm is not None and final_loss_mlm is not None:
            final_loss_mlm = (
                (1 - self.interctc_weight) * final_loss_mlm
                + self.interctc_weight * loss_intermlm
            )

        loss = self._add_optional_loss(final_loss_ctc, final_loss_mlm)
        if loss is None:
            raise RuntimeError("No CTC/MLM loss was produced from intermediate outputs")

        # Collect CTC branch stats
        stats["loss_ctc"] = final_loss_ctc.detach() if final_loss_ctc is not None else None
        stats["cer_ctc"] = cer_ctc[-1] if cer_ctc else None
        stats["cer_mlm"] = cer_mlm[-1] if cer_mlm else None
        stats["loss_mlm"] = final_loss_mlm.detach() if final_loss_mlm is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()
        self.ctc.reset_intermediate_outs()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss and populate intermediate outputs.
        self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        ys_pad_cpu = (
            ys_pad.cpu() if (not self.training and self.error_calculator is not None) else None
        )
        loss_ctc, loss_mlm, cer_ctc, cer_mlm = [], [], [], []
        for pred in self.ctc.intermediate_outs:
            l_ctc, l_mlm, c_ctc, c_mlm = self._calc_single_pred_loss_and_cer(
                pred,
                ys_pad,
                ys_pad_lens,
                encoder_out_lens,
                ys_pad_cpu=ys_pad_cpu,
            )
            loss_ctc.append(l_ctc)
            loss_mlm.append(l_mlm)
            cer_ctc.append(c_ctc)
            cer_mlm.append(c_mlm)

        return loss_ctc, loss_mlm, cer_ctc, cer_mlm
