# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ARECHO autoregressive metric prediction with published checkpoint names."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.amp import autocast
from typeguard import typechecked

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask, th_accuracy
from espnet2.legacy.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet2.legacy.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
)
from espnet2.legacy.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
)
from espnet2.legacy.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.universa.abs_universa import AbsUniversa
from espnet2.universa.ar_universa.universa_beam_search import ARUniVERSABeamSearch
from espnet2.universa.metric_tokenizer.metric_tokenizer import MetricTokenizer


class ARUniversa(AbsUniversa):
    """Encode audio and optional references, then decode metric/value pairs."""

    sequential_metrics = True

    def __init__(
        self,
        # Model Backbone
        input_size: int,
        metric2id: Dict[str, int],
        use_ref_audio: bool = True,
        use_ref_text: bool = True,
        embedding_size: int = 512,
        use_normalize: bool = True,
        audio_encoder_type: str = "transformer",
        audio_encoder_params: Dict[str, Union[float, int, bool, str]] = {
            "num_blocks": 3,
            "attention_heads": 4,
            "linear_units": 2048,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "input_layer": "linear",
            "normalize_before": True,
            "concat_after": False,
            "positionwise_layer_type": "linear",
            "positionwise_conv_kernel_size": 1,
            "layer_drop_rate": 0.0,
            "qk_norm": False,
            "use_flash_attn": False,
        },
        # Metric related
        metric_vocab_size: Optional[int] = None,
        metric_token_info: Optional[Dict[str, Any]] = None,
        metric2type: Optional[Dict[str, str]] = None,
        metric_pad_value: float = -100,
        metric_token_pad_value: int = 0,
        sequential_metrics: bool = True,
        # Text processor
        vocab_size: Optional[int] = None,
        ignore_id: int = -1,
        text_encoder_type: str = "transformer",
        text_encoder_params: Dict[str, Union[float, int, bool, str]] = {
            "num_blocks": 3,
            "attention_heads": 4,
            "linear_units": 2048,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "input_layer": "linear",
            "normalize_before": True,
            "concat_after": False,
            "positionwise_layer_type": "linear",
            "positionwise_conv_kernel_size": 1,
            "layer_drop_rate": 0.0,
            "qk_norm": False,
            "use_flash_attn": False,
        },
        # Attention modules
        cross_attention_type: str = "multihead",
        cross_attention_params: Dict[str, Union[float, int]] = {
            "n_head": 4,
            "dropout_rate": 0.1,
        },
        # Decoder modules
        metric_decoder_params: Dict[str, Union[float, int]] = {
            "num_blocks": 3,
            "attention_heads": 4,
            "linear_units": 2048,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "self_attention_dropout_rate": 0.1,
            "src_attention_dropout_rate": 0.1,
            "input_layer": "embed",
            "use_output_layer": True,
            "normalize_before": True,
            "concat_after": False,
            "layer_drop_rate": 0.0,
            "qk_norm": False,
            "use_flash_attn": False,
        },
        use_rope_pos: bool = False,
        # Other parameters
        lsm_weight: float = 0.0,
        # Pretrained HF Tokenizer may needs custom sym_sos and sym_eos
        sym_sos: str = "<sos>",
        sym_eos: str = "<eos>",
        **kwargs,
    ):
        """Initialize ARECHO with the published architecture and token convention.

        Args:
            input_size (int): Input feature size.
            metric2id (Dict[str, int]): Dictionary mapping metric names to IDs.
            use_ref_audio (bool): Whether to use reference audio.
            use_ref_text (bool): Whether to use reference text.
            embedding_size (int): Embedding size for audio and text encoders.
            use_normalize (bool): Whether to use normalization.
            audio_encoder_type (str): Type of audio encoder.
            audio_encoder_params (Dict[str, Union[float, int, bool, str]]): Parameters
                for audio encoder.
            metric_vocab_size (Optional[int]): Vocabulary size for metrics.
            metric_token_info (Optional[Dict[str, Any]]): Information about metric
                tokens.
            metric2type (Optional[Dict[str, str]]): Legacy config field. Metric types
                are defined by metric_token_info.
            metric_pad_value (float): Legacy config field for regression metrics.
            metric_token_pad_value (int): Padding value for metric tokens.
            sequential_metrics (bool): Whether to use sequential metrics.
            vocab_size (Optional[int]): Vocabulary size for text encoder.
            ignore_id (int): Ignore ID for padding in text encoder.
            text_encoder_type (str): Type of text encoder.
            text_encoder_params (Dict[str, Union[float, int, bool, str]]): Parameters
                for text encoder.
            cross_attention_type (str): Type of cross attention module.
            cross_attention_params (Dict[str, Union[float, int]]): Parameters for cross
                attention module.
            metric_decoder_params (Dict[str, Union[float, int]]): Parameters for metric
                decoder module.
            use_rope_pos (bool): Whether to use RoPE positional encoding.
            lsm_weight (float): Label smoothing weight.
            sym_sos (str): Legacy config field; ARECHO uses SOS ID 2.
            sym_eos (str): Legacy config field; ARECHO uses EOS ID 3.
            **kwargs: Additional parameters.

        """
        super().__init__()

        # Precheck parameters
        if not sequential_metrics:
            raise ValueError(
                "sequential_metrics is required for ar-universa, please set it to True."
            )

        # Initialize parameters
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.metric_vocab_size = metric_vocab_size
        self.ignore_id = ignore_id
        self.use_ref_audio = use_ref_audio
        self.use_ref_text = use_ref_text
        self.embedding_size = embedding_size
        decoder_input_dim = embedding_size
        self.use_normalize = use_normalize
        self.search_module = None
        self.save_token_seq = False

        self.metric2id = metric2id
        self.metric_token_pad_value = metric_token_pad_value
        self.metric_tokenizer = MetricTokenizer(
            metric_token_info, tokenize_metric=list(metric2id.keys())
        )

        # Published ARECHO checkpoints reverse the tokenizer's SOS/EOS names.
        self.sos = 2
        self.eos = 3

        # Initialize audio encoder
        if audio_encoder_type == "transformer":
            self.audio_encoder = TransformerEncoder(
                input_size=input_size,
                output_size=embedding_size,
                **audio_encoder_params,
            )
        else:
            raise ValueError(f"Not supported: {audio_encoder_type}")
        if self.use_normalize:
            self.normalize = UtteranceMVN(norm_means=True, norm_vars=True)

        # Initialize reference audio encoder
        if self.use_ref_audio:
            self.ref_audio_encoder = TransformerEncoder(
                input_size=input_size,
                output_size=embedding_size,
                **audio_encoder_params,
            )
            decoder_input_dim += embedding_size
            if self.use_normalize:
                self.ref_normalize = UtteranceMVN(norm_means=True, norm_vars=True)

        # Initialize text encoder
        if self.use_ref_text:
            self.text_embedding = torch.nn.Embedding(
                vocab_size,
                embedding_size,
            )
            if text_encoder_type == "transformer":
                self.text_encoder = TransformerEncoder(
                    input_size=embedding_size,
                    output_size=embedding_size,
                    **text_encoder_params,
                )
            else:
                raise ValueError(f"Not supported: {text_encoder_type}")
            decoder_input_dim += embedding_size

        # Initialize cross attention
        if cross_attention_type == "multihead":
            self.cross_attention = MultiHeadedAttention(
                n_feat=embedding_size,
                **cross_attention_params,
            )
        else:
            raise ValueError(f"Not supported: {cross_attention_type}")

        if use_rope_pos:
            raise ValueError(
                "use_rope_pos=True is not supported by the released ARECHO checkpoints"
            )
        self.decoder = TransformerDecoder(
            vocab_size=metric_vocab_size,
            encoder_output_size=decoder_input_dim,
            pos_enc_class=PositionalEncoding,
            **metric_decoder_params,
        )

        self.ar_criterion = LabelSmoothingLoss(
            size=metric_vocab_size,
            padding_idx=metric_token_pad_value,
            smoothing=lsm_weight,
            normalize_length=True,
        )

    @typechecked
    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        metrics: Dict[str, torch.Tensor],
        ref_audio: Optional[torch.Tensor] = None,
        ref_audio_lengths: Optional[torch.Tensor] = None,
        ref_text: Optional[torch.Tensor] = None,
        ref_text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate outputs and return the loss tensor.

        Args:
            audio (torch.Tensor): Input audio tensor (B, T).
            audio_lengths (torch.Tensor): Length of audio tensor (B,).
            metrics (torch.Tensor): Metrics tensor Dict[str, tensor (B,)].
            ref_audio (torch.Tensor): Reference audio tensor (B, T).
            ref_audio_lengths (torch.Tensor): Length of reference audio tensor (B,).
            ref_text (torch.Tensor): Reference text tensor (B, U).
            ref_text_lengths (torch.Tensor): Length of reference text tensor (B,).

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
                loss (torch.Tensor): Loss tensor.
                stats (Dict[str, torch.Tensor]): Statistics to be monitored.
                weight (torch.Tensor): Weight tensor.

        """
        assert "metric_token" in metrics, "metric_token is required in metrics"
        assert (
            "metric_token_lengths" in metrics
        ), "metric_token_lengths is required in metrics"
        metric_token, metric_token_lengths = (
            metrics["metric_token"],
            metrics["metric_token_lengths"],
        )

        batch_size = audio.shape[0]
        assert (
            metric_token_lengths.dim() == 1
        ), "metric_token_lengths should be 1D tensor, but received {}".format(
            metric_token_lengths.dim()
        )
        # Check that batch_size is unified
        assert (
            batch_size == audio_lengths.shape[0]
            and batch_size == metric_token.shape[0]
            and batch_size == metric_token_lengths.shape[0]
        ), "mismatch batch size with audio {}, metrics {}, metric_token {}".format(
            audio.shape[0], audio_lengths.shape[0], metric_token.shape[0]
        )

        # for data-parallel
        metric_token = metric_token[:, : metric_token_lengths.max()]
        padding = torch.arange(metric_token.size(1), device=metric_token.device)
        padding = padding.unsqueeze(0) >= metric_token_lengths.to(
            metric_token.device
        ).unsqueeze(1)
        metric_token = metric_token.masked_fill(
            padding | (metric_token == -1), self.metric_token_pad_value
        )

        # 2. Encode audio
        audio_enc, audio_enc_lengths = self.encode(
            audio,
            audio_lengths,
            ref_audio,
            ref_audio_lengths,
            ref_text,
            ref_text_lengths,
        )

        # 3. Metric Decoder
        loss, acc_ar_decoder, value_ar_decoder = self._calc_decoder_loss(
            audio_enc, audio_enc_lengths, metric_token, metric_token_lengths
        )

        stats = {
            "loss_ar_decoder": loss.detach(),
            "acc_ar_decoder": acc_ar_decoder,
            "value_ar_decoder": value_ar_decoder,
            "loss": loss.detach(),
        }

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @typechecked
    def encode(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        ref_audio: Optional[torch.Tensor] = None,
        ref_audio_lengths: Optional[torch.Tensor] = None,
        ref_text: Optional[torch.Tensor] = None,
        ref_text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode references without modifying caller-owned inputs.

        Missing references contribute zero features in their configured slots.
        """
        batch_size = audio.shape[0]

        use_ref_audio = self.use_ref_audio and ref_audio is not None
        use_ref_text = self.use_ref_text and ref_text is not None

        if use_ref_audio and ref_audio_lengths is None:
            raise ValueError("ref_audio_lengths is required with ref_audio")
        if use_ref_text and ref_text_lengths is None:
            raise ValueError("ref_text_lengths is required with ref_text")

        if use_ref_text:
            assert (
                ref_text.shape[0] == batch_size
            ), "mismatch batch size with ref_text {}".format(ref_text.shape[0])
            ref_text = ref_text.masked_fill(ref_text == self.ignore_id, 0)
            # for data-parallel
            ref_text = ref_text[:, : ref_text_lengths.max()]

        # DataParallel keeps global padding even when a shard has shorter inputs.
        audio = audio[:, : audio_lengths.max()]
        if use_ref_audio:
            ref_audio = ref_audio[:, : ref_audio_lengths.max()]

        # 1. Feats normalization
        feats, feats_lengths = audio, audio_lengths
        ref_feats, ref_feats_lengths = ref_audio, ref_audio_lengths
        if use_ref_text:
            ref_text_embed = self.text_embedding(ref_text)
        if self.use_normalize:
            with autocast("cuda", enabled=False):
                feats, feats_lengths = self.normalize(audio.clone(), audio_lengths)
                if use_ref_audio:
                    ref_feats, ref_feats_lengths = self.ref_normalize(
                        ref_audio.clone(), ref_audio_lengths
                    )

        # 2. Encode audio
        audio_enc, audio_enc_lengths, _ = self.audio_encoder(feats, feats_lengths)
        if use_ref_audio:
            ref_audio_enc, ref_audio_enc_lengths, _ = self.ref_audio_encoder(
                ref_feats, ref_feats_lengths
            )
        if use_ref_text:
            ref_text_enc, ref_text_enc_lengths, _ = self.text_encoder(
                ref_text_embed, ref_text_lengths
            )

        # 3. Cross attention
        enc_list = [audio_enc]
        if use_ref_audio:
            ref_audio_mask = (
                ~make_pad_mask(ref_audio_enc_lengths).to(audio_enc.device).unsqueeze(1)
            )
            ref_audio_info = self.cross_attention(
                audio_enc, ref_audio_enc, ref_audio_enc, ref_audio_mask
            )
            enc_list.append(ref_audio_info)
        if use_ref_text:
            ref_text_mask = (
                ~make_pad_mask(ref_text_enc_lengths).to(audio_enc.device).unsqueeze(1)
            )
            ref_text_info = self.cross_attention(
                audio_enc, ref_text_enc, ref_text_enc, ref_text_mask
            )
            enc_list.append(ref_text_info)
        if self.use_ref_audio and not use_ref_audio:
            enc_list.insert(1, torch.zeros_like(audio_enc))
        if self.use_ref_text and not use_ref_text:
            enc_list.append(torch.zeros_like(audio_enc))
        audio_enc = torch.cat(enc_list, dim=-1)

        return audio_enc, audio_enc_lengths

    def _calc_decoder_loss(
        self,
        audio_enc: torch.Tensor,
        audio_enc_lengths: torch.Tensor,
        metric_token: torch.Tensor,
        metric_token_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        """Calculate decoder loss.

        Args:
            audio_enc (torch.Tensor): Encoded audio tensor (B, T, D).
            audio_enc_lengths (torch.Tensor): Length of encoded audio tensor (B,).
            metric_token (torch.Tensor): Metric tokens tensor (B, U).
            metric_token_lengths (torch.Tensor): Length of metric tokens tensor (B,).

        Returns:
            loss_ar_decoder (torch.Tensor): Loss tensor for AR decoder.
            acc_ar_decoder (torch.Tensor): Accuracy tensor for AR decoder.
            value_ar_decoder (torch.Tensor): Value tensor for AR decoder.
        """

        ys_in_pad, ys_out_pad = add_sos_eos(
            metric_token, self.sos, self.eos, self.metric_token_pad_value
        )
        ys_in_lens = metric_token_lengths + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            audio_enc, audio_enc_lengths, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_ar_decoder = self.ar_criterion(decoder_out, ys_out_pad)
        acc_ar_decoder = th_accuracy(
            decoder_out.view(-1, self.metric_vocab_size),
            ys_out_pad,
            ignore_label=self.metric_token_pad_value,
        )
        # An entirely unlabelled batch trains EOS but has no value accuracy.
        acc_value_ar_decoder = (
            th_accuracy(
                decoder_out[:, 1::2].reshape(-1, self.metric_vocab_size),
                ys_out_pad[:, 1::2],
                ignore_label=self.metric_token_pad_value,
            )
            if metric_token_lengths.sum() > 0
            else 0.0
        )

        return loss_ar_decoder, acc_ar_decoder, acc_value_ar_decoder

    @typechecked
    def set_inference(
        self,
        beam_size: int,
        metric_list: List[str],
        skip_meta_label_score: bool,
        save_token_seq: bool = False,
        use_fixed_order: bool = False,
    ) -> None:
        """Set inference mode.

        Args:
            beam_size (int): Beam size for beam search.
            metric_list (List[str]): List of metrics to predict.
            skip_meta_label_score (bool): Whether to skip meta label score.
            save_token_seq (bool): Whether to save token sequence.
            use_fixed_order (bool): Decode metrics in the requested order.
        """
        scorers = {
            "metric_decoder": self.decoder,
        }
        weights = {"metric_decoder": 1.0}

        # Use the same value-token boundaries as training tokenization.
        beam_masking = {}
        for metric_name in metric_list:
            metric_token = self.metric_tokenizer.get_metric_meta_label(metric_name)
            values = self.metric_tokenizer.tokenizer_config[metric_name]
            first_value_index = 1 if isinstance(values[0], str) else 0
            _, start_idx = self.metric_tokenizer.get_token_index(
                metric_name, first_value_index
            )
            _, last_idx = self.metric_tokenizer.get_token_index(
                metric_name, first_value_index + len(values) - 1
            )
            beam_masking[metric_token] = (start_idx, last_idx + 1)

        self.save_token_seq = save_token_seq

        self.search_module = ARUniVERSABeamSearch(
            scorers=scorers,
            weights=weights,
            beam_size=beam_size,
            vocab_size=self.metric_vocab_size,
            sos=self.sos,
            eos=self.eos,
            meta_label_for_search=[
                self.metric_tokenizer.get_metric_meta_label(metric)
                for metric in metric_list
            ],
            token_list=self.metric_tokenizer.get_token_list(),
            skip_meta_label_score=skip_meta_label_score,
            beam_masking=beam_masking,
            use_fixed_order=use_fixed_order,
        )

    @torch.no_grad()
    @typechecked
    def inference(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        ref_audio: Optional[torch.Tensor] = None,
        ref_audio_lengths: Optional[torch.Tensor] = None,
        ref_text: Optional[torch.Tensor] = None,
        ref_text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Return predicted output as a dict.

        Args:
            audio (torch.Tensor): Input audio tensor (B, T).
            audio_lengths (torch.Tensor): Length of audio tensor (B,).
            ref_audio (torch.Tensor): Reference audio tensor (B, T).
            ref_audio_lengths (torch.Tensor): Length of reference audio tensor (B,).
            ref_text (torch.Tensor): Reference text tensor (B, U).
            ref_text_lengths (torch.Tensor): Length of reference text tensor (B,).
            **kwargs: Additional parameters.

        Returns:
            Dict[str, torch.Tensor]: Predicted output.

        """

        if self.search_module is None:
            self.set_inference(
                beam_size=1,
                metric_list=list(self.metric2id.keys()),
                skip_meta_label_score=False,
            )

        # 1. Encode audio
        audio_enc, encoded_lengths = self.encode(
            audio,
            audio_lengths,
            ref_audio,
            ref_audio_lengths,
            ref_text,
            ref_text_lengths,
        )

        assert audio_enc.size(0) == 1, "Inference only supports batch size of 1."

        # 2. Inference
        nbest_hyps = self.search_module.forward(audio_enc[0, : encoded_lengths[0]])

        # NOTE(jiatong): get the top one hypothesis
        assert len(nbest_hyps) > 0, "nbest_hyps should not be empty"
        assert len(nbest_hyps[0].yseq) > 0, "nbest_hyps[0].yseq should not be empty"
        pred_metrics = nbest_hyps[0].yseq

        # 3. Decorate the predicted metrics
        pred_metrics = self.metric_tokenizer.tokenseq2metric(
            pred_metrics, return_dict=True
        )

        if self.save_token_seq:
            pred_metrics["token_seq"] = [[int(token) for token in nbest_hyps[0].yseq]]

        pred_metrics["use_tokenizer_metrics"] = True
        pred_metrics["sequential_metrics"] = True
        pred_metrics["encoded_feat"] = audio_enc
        return pred_metrics
