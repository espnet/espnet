# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""UniversaBase related modules."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.amp import autocast
from typeguard import typechecked

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.legacy.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
)
from espnet2.spk.pooling.chn_attn_stat_pooling import ChnAttnStatPooling
from espnet2.spk.pooling.mean_pooling import MeanPooling
from espnet2.spk.projector.xvector_projector import XvectorProjector
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.universa.abs_universa import AbsUniversa
from espnet2.universa.base.loss import masked_l1_loss, masked_mse_loss


class UniversaBase(AbsUniversa):
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
        # MultiTask predictors
        pooling_type: str = "mean",
        pooling_params: Dict[str, Union[float, int, bool, str]] = {},
        projector_type: str = "linear",
        projector_params: Dict[str, Union[float, int, bool, str]] = {},
        multi_branch: bool = False,
        use_mse: bool = False,
        use_l1: bool = True,
        metric_pad_value: float = -100,
        loss_weights: Optional[Dict[Union[str, int], float]] = None,
        **kwargs,
    ):
        """Initialize UniversaBase module.

        Args:
            input_size (int): Input dimension.
            metric2id (Dict[str, int]): Metric to ID mapping.
            vocab_size (Optional[int]): Number of vocabulary.
            ignore_id (int): Ignore ID.
            use_ref_audio (bool): Whether to use reference audio.
            use_ref_text (bool): Whether to use reference text.
            embedding_size (int): Embedding size.
            use_normalize (bool): Whether to normalize input features.
            audio_encoder_type (str): Audio encoder type.
            audio_encoder_params (Dict[str, Sequence]): Audio encoder parameters.
            text_encoder_type (str): Text encoder type.
            text_encoder_params (Dict[str, Sequence]): Text encoder parameters.
            cross_attention_type (str): Cross attention type.
            cross_attention_params (Dict[str, Sequence]): Cross attention parameters.
            pooling_type (str): Pooling type.
            pooling_params (Dict[str, Sequence]): Pooling parameters.
            projector_type (str): Projector type.
            projector_params (Dict[str, Sequence]): Projector parameters.
            multi_branch (bool): Whether to use multi-branch pooling and projectors.
            use_mse (bool): Whether to use MSE loss.
            use_l1 (bool): Whether to use L1 loss.
            metric_pad_value (float): Metric padding value.
            loss_weights (Optional[Dict[str, float]]): Loss weights.

        """
        super().__init__()

        if (
            not metric2id
            or any(type(i) is not int for i in metric2id.values())
            or set(metric2id.values()) != set(range(len(metric2id)))
        ):
            raise ValueError("metric2id must contain unique integer IDs from 0 to N-1")
        if use_ref_text and (vocab_size is None or vocab_size <= 0):
            raise ValueError("vocab_size must be positive when use_ref_text=True")

        # Initialize parameters
        self.input_size = input_size
        self.metric_size = len(metric2id)
        self.metric2id = metric2id
        self.id2metric = {v: k for k, v in metric2id.items()}
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.use_ref_audio = use_ref_audio
        self.use_ref_text = use_ref_text
        self.embedding_size = embedding_size
        pooling_dim = embedding_size
        self.use_normalize = use_normalize
        self.use_mse = use_mse
        self.use_l1 = use_l1
        assert (
            self.use_mse or self.use_l1
        ), "At least one loss function should be enabled"
        self.metric_pad_value = metric_pad_value

        # Legacy configs may index weights by ID; new configs can use metric names.
        if loss_weights is None:
            self.loss_weights = dict.fromkeys(range(self.metric_size), 1.0)
        elif set(loss_weights) == set(metric2id):
            self.loss_weights = {i: loss_weights[name] for name, i in metric2id.items()}
        elif set(loss_weights) == set(range(self.metric_size)):
            self.loss_weights = dict(loss_weights)
        else:
            raise ValueError("loss_weights must cover all metric names or integer IDs")

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
            if audio_encoder_type == "transformer":
                self.ref_audio_encoder = TransformerEncoder(
                    input_size=input_size,
                    output_size=embedding_size,
                    **audio_encoder_params,
                )
            else:
                raise ValueError(f"Not supported: {audio_encoder_type}")
            pooling_dim += embedding_size
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
            pooling_dim += embedding_size

        # Initialize cross attention
        if cross_attention_type == "multihead":
            self.cross_attention = MultiHeadedAttention(
                n_feat=embedding_size,
                **cross_attention_params,
            )
        else:
            raise ValueError(f"Not supported: {cross_attention_type}")

        self.multi_branch = multi_branch

        if self.multi_branch:
            self.pooling = torch.nn.ModuleList()
            self.projector = torch.nn.ModuleList()
            for i in range(self.metric_size):
                # Initialize pooling
                if pooling_type == "mean":
                    self.pooling.append(
                        MeanPooling(input_size=pooling_dim, **pooling_params)
                    )
                elif pooling_type == "channel_attention":
                    self.pooling.append(
                        ChnAttnStatPooling(input_size=pooling_dim, **pooling_params)
                    )
                else:
                    raise ValueError(f"Not supported: {pooling_type}")

                projector_input = self.pooling[-1].output_size()
                # Initialize projector
                if projector_type == "linear":
                    self.projector.append(
                        torch.nn.Linear(
                            projector_input,
                            1,
                            **projector_params,
                        )
                    )
                elif projector_type == "xvector":
                    self.projector.append(
                        XvectorProjector(
                            projector_input,
                            1,
                            **projector_params,
                        )
                    )
                else:
                    raise ValueError(f"Not supported: {projector_type}")
        else:
            # Initialize pooling
            if pooling_type == "mean":
                self.pooling = MeanPooling(input_size=pooling_dim, **pooling_params)
            elif pooling_type == "channel_attention":
                self.pooling = ChnAttnStatPooling(
                    input_size=pooling_dim, **pooling_params
                )
            else:
                raise ValueError(f"Not supported: {pooling_type}")

            projector_input = self.pooling.output_size()

            # Initialize projector
            if projector_type == "linear":
                self.projector = torch.nn.Linear(
                    projector_input,
                    self.metric_size,
                    **projector_params,
                )
            elif projector_type == "xvector":
                self.projector = XvectorProjector(
                    projector_input,
                    self.metric_size,
                    **projector_params,
                )
            else:
                raise ValueError(f"Not supported: {projector_type}")

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
        batch_size = audio.shape[0]

        # 1. Prepare metrics
        final_metrics = []
        for i in range(self.metric_size):
            if self.id2metric[i] not in metrics:
                final_metrics.append(
                    torch.zeros(batch_size, dtype=audio.dtype).to(audio.device)
                    + self.metric_pad_value
                )
            else:
                final_metrics.append(metrics[self.id2metric[i]].to(audio.device))
        final_metrics = torch.stack(final_metrics, dim=-1)

        # 2. Encode audio
        audio_enc, audio_enc_lengths = self.encode(
            audio,
            audio_lengths,
            ref_audio,
            ref_audio_lengths,
            ref_text,
            ref_text_lengths,
        )

        pred_metrics = self._predict_metrics(audio_enc, audio_enc_lengths)
        loss = 0.0
        stats = {}
        if self.multi_branch:
            for i in range(self.metric_size):
                pred_metric = pred_metrics[i]
                metric_loss = 0.0
                # NOTE(jiatong): we use > instead of != to handle the case
                # where the metric_pad_value is not 0
                final_metric_mask = final_metrics[:, i] > (self.metric_pad_value + 1e-6)
                if self.use_mse:
                    metric_mse_loss = masked_mse_loss(
                        pred_metric.squeeze(-1), final_metrics[:, i], final_metric_mask
                    )
                    metric_loss = metric_loss + metric_mse_loss
                    stats[self.id2metric[i] + "_mse"] = metric_mse_loss.detach()
                if self.use_l1:
                    metric_l1_loss = masked_l1_loss(
                        pred_metric.squeeze(-1), final_metrics[:, i], final_metric_mask
                    )
                    metric_loss = metric_loss + metric_l1_loss
                    stats[self.id2metric[i] + "_l1"] = metric_l1_loss.detach()
                metric_loss = metric_loss * self.loss_weights[i]
                stats[self.id2metric[i] + "_overall"] = metric_loss.detach()
                loss = loss + metric_loss
            stats["loss"] = loss.detach()
        else:
            pred_metric = pred_metrics
            final_metric_mask = final_metrics > self.metric_pad_value + 1e-6
            weights = pred_metric.new_tensor(
                [self.loss_weights[i] for i in range(self.metric_size)]
            )
            if self.use_mse:
                metric_mse_loss = masked_mse_loss(
                    pred_metric, final_metrics, final_metric_mask, weights
                )
                loss = loss + metric_mse_loss
                stats["mse"] = metric_mse_loss.detach()
            if self.use_l1:
                metric_l1_loss = masked_l1_loss(
                    pred_metric, final_metrics, final_metric_mask, weights
                )
                loss = loss + metric_l1_loss
                stats["l1"] = metric_l1_loss.detach()
            stats["loss"] = loss.detach()

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
        batch_size = audio.shape[0]

        use_ref_audio = self.use_ref_audio and ref_audio is not None
        use_ref_text = self.use_ref_text and ref_text is not None

        if use_ref_text:
            assert (
                ref_text.shape[0] == batch_size
            ), "mismatch batch size with ref_text {}".format(ref_text.shape[0])
            ref_text = ref_text.masked_fill(ref_text == self.ignore_id, 0)
            # for data-parallel
            ref_text = ref_text[:, : ref_text_lengths.max()]

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
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Return predicted output as a dict.

        Args:
            audio (torch.Tensor): Input audio tensor (B, T).
            audio_lengths (torch.Tensor): Length of audio tensor (B,).

        Returns:
            Dict[str, torch.Tensor]: Predicted output.

        """

        # 1. Encode audio
        audio_enc, audio_enc_lengths = self.encode(
            audio,
            audio_lengths,
            ref_audio,
            ref_audio_lengths,
            ref_text,
            ref_text_lengths,
        )

        pred_metrics = self._predict_metrics(audio_enc, audio_enc_lengths)
        pred_metrics = self._inference_decoration(pred_metrics)
        pred_metrics["encoded_feat"] = audio_enc
        return pred_metrics

    def _predict_metrics(
        self, audio_enc: torch.Tensor, audio_enc_lengths: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Pool and project encoded audio without detaching training gradients."""
        if self.multi_branch:
            pred_metrics = []
            for i in range(self.metric_size):
                pooling_output = self.pooling[i](
                    audio_enc.permute(0, 2, 1), feat_lengths=audio_enc_lengths
                )
                with autocast("cuda", enabled=False):
                    # skip numeric stability with float16
                    pred_metric = self.projector[i](pooling_output)
                pred_metrics.append(pred_metric)
        else:
            pooling_output = self.pooling(
                audio_enc.permute(0, 2, 1), feat_lengths=audio_enc_lengths
            )
            with autocast("cuda", enabled=False):
                # skip numeric stability with float16
                pred_metrics = self.projector(pooling_output)
        return pred_metrics

    @typechecked
    def _inference_decoration(
        self,
        pred_metrics: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Decorate the predicted metrics.

        Args:
            pred_metrics (torch.Tensor, List[torch.Tensor]): Predicted metrics tensor.

        Returns:
            Dict[str, Union[np.ndarray, torch.Tensor]]: Decorated predicted metrics.

        """
        if self.multi_branch:
            results = {
                self.id2metric[i]: pred_metrics[i].squeeze(-1).detach().cpu().numpy()
                for i in range(self.metric_size)
            }
        else:
            results = {
                self.id2metric[i]: pred_metrics[:, i].detach().cpu().numpy()
                for i in range(self.metric_size)
            }
        return results
