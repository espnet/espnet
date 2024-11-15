# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""UniversaBase related modules."""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet2.universa.abs_universa import AbsUniversa
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.spk.pooling.mean_pooling import MeanPooling
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention


class UniversaBase(AbsUniversa):
    def __init__(
        self,
        # Model Backbone
        input_size: int,
        metric2id: Dict[str, int],
        use_ref_audio: bool = True,
        use_ref_text: bool = True,
        embedding_size: int = 512,
        audio_encoder_type: str = "transformer",
        audio_encoder_params: Dict[str, Sequence] = {
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
        text_encoder_type: str = "transformer",
        text_encoder_params: Dict[str, Sequence] = {
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
        cross_attention_params: Dict[str, Sequence] = {
            "n_head": 4,
            "dropout_rate": 0.1,
        },
        # MultiTask predictors
        pooling_type: str = "mean",
        pooling_params: Dict[str, Sequence] = {},
        projector_type: str = "linear",
        projector_params: Dict[str, Sequence] = {},
        multi_branch: bool = False,
        **kwargs,
    ):
        """Initialize UniversaBase module.

        Args:
            input_size (int): Input dimension.
            metric_size (int): Number of metrics.
            vocab_size (Optional[int]): Number of vocabulary.
            use_ref_audio (bool): Whether to use reference audio.
            use_ref_text (bool): Whether to use reference text.
            embedding_size (int): Embedding size.
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

        """
        super().__init__()

        # Initialize parameters
        self.input_size = input_size
        self.metric_size = len(metric2id)
        self.metric2id = metric2id
        self.use_ref_audio = use_ref_audio
        self.use_ref_text = use_ref_text
        self.embedding_size = embedding_size
        pooling_dim = embedding_size

        # Initialize audio encoder
        if audio_encoder_type == "transformer":
            self.audio_encoder = TransformerEncoder(
                input_size=input_size,
                output_size=embedding_size,
                **audio_encoder_params,
            )
        else:
            raise ValueError(f"Not supported: {audio_encoder_type}")
        
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
            self.pooling = torch.nn.ModuleDict()
            self.projector = torch.nn.ModuleDict()
            for i in range(self.metric_size):
                # Initialize pooling
                if pooling_type == "mean":
                    self.pooling[str(i)] = MeanPooling(input_size=pooling_dim, **pooling_params)
                else:
                    raise ValueError(f"Not supported: {pooling_type}")

                # Initialize projector
                if projector_type == "linear":
                    self.projector[str(i)] = torch.nn.Linear(
                        pooling_dim,
                        1,
                        **projector_params,
                    )
                else:
                    raise ValueError(f"Not supported: {projector_type}")
        else:
            # Initialize pooling
            if pooling_type == "mean":
                self.pooling = MeanPooling(input_size=pooling_dim, **pooling_params)
            else:
                raise ValueError(f"Not supported: {pooling_type}")

            # Initialize projector
            if projector_type == "linear":
                self.projector = torch.nn.Linear(
                    pooling_dim,
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
        metrics: torch.Tensor,
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
            metrics (torch.Tensor): Metrics tensor (B, C).
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
        pass

    @typechecked
    def inference(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return predicted output as a dict.

        Args:
            audio (torch.Tensor): Input audio tensor (B, T).
            audio_lengths (torch.Tensor): Length of audio tensor (B,).

        Returns:
            Dict[str, torch.Tensor]: Predicted output.

        """
        pass
        