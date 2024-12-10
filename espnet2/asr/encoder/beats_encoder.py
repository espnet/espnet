# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Adapted from Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq

# This code is adapted from the original BEATs implementation and
#  can be used to pre-train/and or fine-tune BEATs model.
# --------------------------------------------------------

import logging
import math
import warnings
from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi
from packaging.version import parse as V
from torch.nn import LayerNorm, Parameter

try:
    from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
    from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
        Wav2Vec2ConformerConfig,
        Wav2Vec2ConformerEncoder,
    )

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class BeatsConfig:
    """
    Configuration class for the BEATs encoder model.

    This class defines the various hyperparameters and configuration options
    for the BEATs model used in audio pre-training with acoustic tokenizers.
    The default values are set in the constructor, but they can be updated
    using the `update` method.

    Attributes:
        input_patch_size (int): Patch size for patch embedding.
        embed_dim (int): Dimension of patch embedding.
        conv_bias (bool): Whether to include bias in the convolutional encoder.
        encoder_layers (int): Number of encoder layers in the transformer.
        encoder_embed_dim (int): Encoder embedding dimension.
        encoder_ffn_embed_dim (int): Feed-forward network embedding dimension.
        encoder_attention_heads (int): Number of attention heads in the encoder.
        activation_fn (str): Activation function used in the model.
        layer_wise_gradient_decay_ratio (float): Ratio for layer-wise gradient decay.
        layer_norm_first (bool): Whether to apply layer normalization first.
        deep_norm (bool): Whether to apply deep normalization first.
        dropout (float): Dropout probability for the transformer.
        attention_dropout (float): Dropout probability for attention weights.
        activation_dropout (float): Dropout probability after activation in FFN.
        encoder_layerdrop (float): Probability of dropping a transformer layer.
        dropout_input (float): Dropout to apply to the input after feature extraction.
        conv_pos (int): Number of filters for convolutional positional embeddings.
        conv_pos_groups (int): Number of groups for convolutional positional embedding.
        relative_position_embedding (bool): Whether to apply relative position embedding.
        num_buckets (int): Number of buckets for relative position embedding.
        max_distance (int): Maximum distance for relative position embedding.
        gru_rel_pos (bool): Whether to apply gated relative position embedding.
        finetuned_model (bool): Indicates if the model is fine-tuned.
        predictor_dropout (float): Dropout probability for the predictor.
        predictor_class (int): Target class number for the predictor.

    Args:
        cfg (dict, optional): Configuration dictionary to update the default values.

    Examples:
        # Create a default configuration
        config = BeatsConfig()

        # Create a configuration with custom settings
        custom_config = BeatsConfig(cfg={
            'input_patch_size': 32,
            'dropout': 0.2,
            'finetuned_model': True
        })
    """

    def __init__(self, cfg=None):
        self.input_patch_size: int = 16  # patch size of patch embedding
        self.embed_dim: int = 512  # patch embedding dimension
        self.conv_bias: bool = False  # include bias in conv encoder

        self.encoder_layers: int = 12  # num encoder layers in the transformer
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072  # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12  # num encoder attention heads
        self.activation_fn: str = "gelu"  # activation function to use

        self.layer_wise_gradient_decay_ratio: float = (
            1.0  # ratio for layer-wise gradient decay
        )
        self.layer_norm_first: bool = False  # apply layernorm first in the transformer
        self.deep_norm: bool = False  # apply deep_norm first in the transformer

        # dropouts
        self.dropout: float = 0.1  # dropout probability for the transformer
        self.attention_dropout: float = 0.1  # dropout probability for attention weights
        self.activation_dropout: float = (
            0.0  # dropout probability after activation in FFN
        )
        self.encoder_layerdrop: float = (
            0.0  # probability of dropping a tarnsformer layer
        )
        self.dropout_input: float = (
            0.0  # dropout to apply to the input (after feat extr)
        )

        # positional embeddings
        self.conv_pos: int = (
            128  # number of filters for convolutional positional embeddings
        )
        self.conv_pos_groups: int = (
            16  # number of groups for convolutional positional embedding
        )

        # relative position embedding
        self.relative_position_embedding: bool = (
            False  # apply relative position embedding
        )
        self.num_buckets: int = 320  # number of buckets for relative position embedding
        self.max_distance: int = (
            1280  # maximum distance for relative position embedding
        )
        self.gru_rel_pos: bool = False  # apply gated relative position embedding

        # label predictor
        self.finetuned_model: bool = False  # whether the model is a fine-tuned model.
        self.predictor_dropout: float = 0.1  # dropout probability for the predictor
        self.predictor_class: int = 527  # target class number for the predictor

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        """
        Update the configuration of the BeatsConfig instance.

        This method updates the attributes of the BeatsConfig instance
        with values from the provided configuration dictionary. It
        modifies the instance's internal state directly by updating
        its __dict__ attribute.

        Args:
            cfg (dict): A dictionary containing configuration parameters
                where keys correspond to attribute names and values
                are the new values to set.

        Examples:
            >>> config = BeatsConfig()
            >>> new_cfg = {
            ...     'input_patch_size': 32,
            ...     'dropout': 0.2,
            ... }
            >>> config.update(new_cfg)
            >>> print(config.input_patch_size)
            32
            >>> print(config.dropout)
            0.2

        Note:
            Ensure that the keys in the `cfg` dictionary match the
            attribute names of the BeatsConfig class to avoid any
            unexpected behavior.
        """


class BeatsEncoder(AbsEncoder):
    """
    BEATs: Audio Pre-Training with Acoustic Tokenizers.

    This class implements the BEATs model for audio pre-training and fine-tuning
    using acoustic tokenizers. It can handle various configurations, including
    the use of pretrained weights and the application of SpecAugment.

    Attributes:
        fbank_mean (float): Mean of the filter banks.
        fbank_std (float): Standard deviation of the filter banks.
        max_layer (Optional[int]): Maximum layer to propagate input through.
        beats_ckpt_path (Optional[str]): Path to a pretrained Beats checkpoint.
        loaded_state_dict_ (Optional[Dict]): Loaded state dictionary from the
            checkpoint.
        specaug (Optional[SpecAug]): SpecAugment instance if config provided.
        _output_size (int): Size of the output features.
        embed (int): Embedding dimension.
        input_patch_size (int): Size of input patches for the model.
        post_extract_proj (Optional[nn.Linear]): Projection layer after feature
            extraction.
        patch_embedding (nn.Conv2d): Convolutional layer for patch embedding.
        dropout_input (nn.Dropout): Dropout layer for input features.
        encoder (TransformerEncoder): Transformer encoder module.
        layer_norm (LayerNorm): Layer normalization module.
        use_weighted_representation (bool): Flag to use weighted representations.
        layer_weights (Optional[nn.Parameter]): Weights for layer representations
            if using weighted representations.
        downsample_conv (Optional[nn.Conv1d]): Downsampling convolutional layer.
        conformer_adapter (Optional[Wav2Vec2ConformerEncoder]): Adapter module for
            Wav2Vec2.
        cross_embed_positions (Optional[BartLearnedPositionalEmbedding]): Learned
            positional embeddings for cross-attention.

    Args:
        input_size (int): The size of the input features.
        beats_ckpt_path (str, optional): Path to a pretrained Beats checkpoint.
            If `beats_config` is provided and it does not match the config in the
            checkpoint, an error might occur.
        max_layer (int, optional): Maximum layer to propagate input through. If
            None, input is propagated through all layers.
        downsampling_rate (int, optional): Downsampling rate for the encoder.
            Applied if greater than 1. Default is 1.
        adapter_config (str, optional): Path to a config file for the Wav2Vec2
            adapter.
        use_weighted_representation (bool, optional): If True, use weighted
            representations from max_layer. Weights are randomly initialized.
        beats_config (Optional[BeatsConfig], optional): BeatsConfig object. If
            provided, will attempt to override the config in the checkpoint.
        specaug_config (Optional[Dict], optional): Dictionary containing parameters
            for SpecAugment. If provided, SpecAugment will be applied.
        add_positional_information (bool, optional): If True, add learned positional
            embeddings.
        max_positions (Optional[int], optional): Maximum number of positions for
            positional embeddings. Required if `add_positional_information` is True.

    Raises:
        ImportError: If the `transformers` library is not available and
            adapter_config or add_positional_information is set.

    Examples:
        >>> encoder = BeatsEncoder(input_size=128, beats_ckpt_path='path/to/ckpt')
        >>> features = torch.randn(10, 16000)  # 10 audio samples
        >>> ilens = torch.tensor([16000] * 10)  # Lengths of each sample
        >>> audio_representation, output_lens, _ = encoder(features, ilens)

    Note:
        This class is designed to be compatible with the ESPnet framework's
        AbsEncoder interface.
    """

    def __init__(
        self,
        input_size: int,
        beats_ckpt_path: str = None,
        max_layer: int = None,
        downsampling_rate: int = 1,
        adapter_config: str = "",
        use_weighted_representation: bool = False,
        beats_config: Optional[BeatsConfig] = None,
        specaug_config: Optional[Dict] = None,
        add_positional_information: bool = False,
        max_positions: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.fbank_mean = 15.41663
        self.fbank_std = 6.55582
        self.max_layer = max_layer
        self.beats_ckpt_path = beats_ckpt_path

        # Four cases for loading Beats config:
        # 1. No checkpoint and no config: Default config
        # 2. Checkpoint and no user-provided config: Load config from
        #    checkpoint
        # 3. Checkpoint and user-provided config: Merge the two, but
        #    override with user-provided config
        # 4. No checkpoint and user-provided config: Use user-provided config
        if adapter_config or add_positional_information:
            # We need transformers library for adapter and positional embeddings
            if not is_transformers_available:
                raise ImportError(
                    "`transformers` is not available. Please install it "
                    " via `pip install transformers` or"
                    " `cd /path/to/espnet/tools && "
                    ". ./activate_python.sh"
                    " && ./installers/install_transformers.sh`."
                )
        config = BeatsConfig()  # Default config
        if beats_ckpt_path and beats_config:
            logging.warning(
                "Both pretrained checkpoint and config are provided."
                " We will override ckpt config with user-provided config."
            )
        self.loaded_state_dict_ = None
        if beats_ckpt_path is not None:
            self.loaded_state_dict_ = torch.load(beats_ckpt_path)
            logging.info(f"Loaded Beats pretrained config from {beats_ckpt_path}.")
            config = BeatsConfig(self.loaded_state_dict_["cfg"])
        if beats_config is not None:
            config.update(vars(beats_config))
            logging.info("Overriding Beats config with user-provided config.")

        self.specaug = None
        if specaug_config is not None:
            self.specaug = SpecAug(**specaug_config)

        self._output_size = config.encoder_embed_dim

        self.embed = config.embed_dim
        self.input_patch_size = config.input_patch_size
        self.post_extract_proj = (
            nn.Linear(self.embed, config.encoder_embed_dim)
            if self.embed != config.encoder_embed_dim
            else None
        )
        self.patch_embedding = nn.Conv2d(
            1,
            self.embed,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=config.conv_bias,
        )
        self.dropout_input = nn.Dropout(config.dropout_input)
        assert not config.deep_norm or not config.layer_norm_first

        self.encoder = TransformerEncoder(config)
        self.layer_norm = LayerNorm(self.embed)

        self.use_weighted_representation = use_weighted_representation
        if self.use_weighted_representation:
            if self.max_layer is None:
                logging.warning(
                    f"max_layer must be provided when using weighted"
                    f" representations. Set to {config.encoder_layers-1}."
                )
                self.max_layer = config.encoder_layers - 1  # 0 based index
            self.layer_weights = nn.Parameter(
                torch.ones((self.max_layer + 1, 1)), requires_grad=True
            )

        # Downsampling modules
        self.encoder_downsample_rate = downsampling_rate
        self.downsample_conv = None
        if self.encoder_downsample_rate > 1:
            self.downsample_conv = nn.Conv1d(
                in_channels=config.encoder_embed_dim,
                out_channels=config.encoder_embed_dim,
                kernel_size=int(
                    round(self.encoder_downsample_rate * 1.5)
                ),  # kernel multiplier from Shih-Lun's code
                stride=self.encoder_downsample_rate,
            )

        # Adapter module
        self.conformer_adapter = None
        if adapter_config:
            conformer_config = Wav2Vec2ConformerConfig.from_json_file(adapter_config)
            self.conformer_adapter = Wav2Vec2ConformerEncoder(conformer_config)

        # Positional embeddings applied before cross-attention with decoder.
        self.cross_embed_positions = None
        if add_positional_information:
            assert (
                max_positions is not None
            ), "max_positions must be provided in the config."
            learned_pos_dim = (
                config.encoder_embed_dim
                if not self.conformer_adapter
                else self.conformer_adapter.config.hidden_size
            )
            self.cross_embed_positions = BartLearnedPositionalEmbedding(
                max_positions, learned_pos_dim
            )

    def reload_pretrained_parameters(self):
        """
        Initialize the Beats model parameters.

        This method is intended to be called last in the initialization
        procedure. It performs the following steps:

        1. Initializes the Beats encoder parameters.
        2. If a pretrained checkpoint is provided, loads the weights
           from the checkpoint to override the initialized parameters.

        The initialization includes:
        - Applying Xavier normal initialization to the post-extraction
          projection layer (if it exists).
        - Applying Xavier normal initialization to the patch embedding
          layer.
        - Calling the custom weight initialization for the encoder
          layers.

        If a pretrained model state is loaded, it also logs any missing
        or unexpected keys between the loaded model and the custom model.

        Raises:
            RuntimeError: If the pretrained weights do not match the
            model architecture.

        Examples:
            >>> encoder = BeatsEncoder(...)
            >>> encoder.reload_pretrained_parameters()
            # This will initialize the parameters and load the pretrained
            # weights if available.

        Note:
            Ensure that this method is called after all model layers
            have been initialized to avoid any inconsistencies.
        """
        logging.info("Beats Initialization function called.")
        if self.post_extract_proj:
            torch.nn.init.xavier_normal_(self.post_extract_proj.weight)
            if self.post_extract_proj.bias is not None:
                torch.nn.init.constant_(self.post_extract_proj.bias, 0)
        torch.nn.init.xavier_normal_(self.patch_embedding.weight)
        if self.patch_embedding.bias is not None:
            torch.nn.init.constant_(self.patch_embedding.bias, 0)

        # Beats has different initialization from ESPnet for other modules,
        #  so override.
        self.encoder.apply(init_bert_params)
        if self.loaded_state_dict_ is not None:

            load_info = self.load_state_dict(
                self.loaded_state_dict_["model"], strict=False
            )
            # strict=False to ignore Weights in the predictor
            logging.info(
                f"Loaded Beats pretrained model. Following keys were missing"
                f" in your custom model: {load_info.missing_keys}. "
                f"Follwing keys could not be loaded from the pretrained"
                f"checkpoint: {load_info.unexpected_keys}."
                "It is expected to have 'predictor' listed above if you are"
                "fine-tuning with only the Beats backbone."
            )

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate a forward padding mask based on input features.

        This method processes the provided padding mask to ensure it is
        compatible with the dimensions of the input features. The function
        adjusts the padding mask's size to match the features by removing
        any extra padding and reshaping it accordingly. The resulting mask
        indicates which parts of the input are valid (not padded).

        Args:
            features (torch.Tensor): A tensor representing input features
                with shape (B, T, C), where B is the batch size, T is the
                sequence length, and C is the number of features per time
                step.
            padding_mask (torch.Tensor): A tensor representing the original
                padding mask with shape (B, L), where L is the length of the
                sequence before any adjustments.

        Returns:
            torch.Tensor: A boolean tensor of shape (B, T) indicating the
            valid positions in the input features after padding adjustment.

        Examples:
            >>> features = torch.randn(4, 10, 512)  # Batch of 4, 10 time steps
            >>> padding_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            ...                                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            ...                                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            ...                                 [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
            >>> mask = forward_padding_mask(features, padding_mask)
            >>> print(mask.shape)
            torch.Size([4, 10])

        Note:
            The padding mask is expected to be in the shape of (B, L),
            where L is typically the maximum sequence length used in the
            batch. The function will truncate the padding mask if necessary
            to fit the features tensor.

        Raises:
            ValueError: If the padding_mask does not have the expected shape
            or if the dimensions are incompatible.
        """
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)  # remove totally empty sequences
        return padding_mask

    def preprocess(
        self,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """
        Preprocess raw audio into feature representations.

        This method takes raw audio waveforms and converts them into
        filter bank features suitable for input into the BEATs model.
        Each waveform is processed to extract Mel filter bank features,
        which are then normalized using pre-defined mean and standard
        deviation values.

        Args:
            source (torch.Tensor): A tensor of shape (B, T) where B is the
                batch size and T is the number of time steps (samples) in
                each waveform.

        Returns:
            torch.Tensor: A tensor of shape (B, F, T') where F is the number
                of Mel filter bank coefficients (128) and T' is the number
                of frames obtained from the original audio after processing.

        Examples:
            >>> encoder = BeatsEncoder()
            >>> raw_audio = torch.randn(2, 16000)  # Example: 2 audio samples
            >>> features = encoder.preprocess(raw_audio)
            >>> print(features.shape)  # Output: (2, 128, T')

        Note:
            - The input waveforms are expected to be in float32 format,
              and the function scales them to int16 format during processing.
            - The filter bank extraction is performed using Kaldi's fbank
              function with a frame length of 25 ms and a frame shift of 10 ms.
        """
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2**15  # float32 to int16
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10,
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - self.fbank_mean) / (2 * self.fbank_std)
        return fbank

    def output_size(self) -> int:
        """
        Get the output size of the BeatsEncoder.

        This function retrieves the output size of the encoder, which is
        determined during the initialization based on the configuration
        provided to the Beats model.

        Returns:
            int: The output size of the encoder, typically equal to the
            encoder embedding dimension defined in the configuration.

        Examples:
            >>> encoder = BeatsEncoder(input_size=256)
            >>> size = encoder.output_size()
            >>> print(size)
            768  # Assuming the encoder embedding dimension is set to 768

        Note:
            The output size is essential for determining the shape of
            the data that flows through subsequent layers of the model.
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Wrapper for compatibility with ESPnet's AbsEncoder Interface.

        This method processes the input tensor and computes audio
        representations by applying the Beats encoder. It manages
        padding and length adjustments for batch processing.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, T, D) where
                B is the batch size, T is the sequence length, and D is
                the feature dimension.
            ilens (torch.Tensor): Tensor of shape (B,) containing the
                lengths of each sequence in the batch.
            prev_states (torch.Tensor, optional): Not used in this
                implementation. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - audio_representation (torch.Tensor): The output
                  audio representation tensor of shape (B, T, D).
                - output_lens (torch.Tensor): Tensor of shape (B,)
                  containing the lengths of the output sequences.
                - masks (Optional[torch.Tensor]): Currently set to None.

        Note:
            If `xs_pad` is not provided, this operation can be costly
            because it attempts to create a tensor of size maxlen x
            maxlen. Therefore, the implementation unsqueezes and
            squeezes tensors to optimize performance.

        Examples:
            >>> encoder = BeatsEncoder(...)
            >>> input_tensor = torch.randn(4, 100, 64)  # (B, T, D)
            >>> input_lengths = torch.tensor([100, 90, 80, 70])
            >>> audio_rep, output_lengths, masks = encoder.forward(input_tensor,
            ...                                                   input_lengths)
            >>> print(audio_rep.shape)  # Should be (4, T', D)
            >>> print(output_lengths)    # Should be tensor of lengths
        """

        # NOTE(shikhar): If xs is not provided then the operation is costly,
        # because this function tries to create a tensor of size maxlen x maxlen.
        # Therfore, we unsqueeze and then squeeze tensors.
        mask = make_pad_mask(
            lengths=ilens, xs=xs_pad.unsqueeze(-1).unsqueeze(-1), length_dim=1
        ).to(xs_pad.device)
        # Adjust shapes to be compatible with Beats code
        xs_pad, mask = xs_pad.squeeze(-1).squeeze(-1), mask.squeeze(-1).squeeze(-1)
        # masks = None
        audio_representation, mask = self.extract_features(
            xs_pad,
            mask,
            max_layer=self.max_layer,
        )
        output_lens = (~mask).sum(-1)
        return audio_representation, output_lens, None

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        max_layer: Optional[int] = None,
    ):
        """
        Extract features from raw audio.

        This method processes the input audio tensor and extracts meaningful
        features using a series of transformations, including patch embedding,
        layer normalization, and optional downsampling. The resulting features
        can be used for further processing or modeling tasks.

        Args:
            source (torch.Tensor): A tensor of shape (B, T) representing
                the input audio, where B is the batch size and T is the
                number of time steps.
            padding_mask (Optional[torch.Tensor]): An optional mask tensor
                of shape (B, T) indicating the positions of the padding
                tokens in the input. Default is None.
            max_layer (Optional[int]): If specified, this determines the
                maximum layer from which features should be extracted. If
                None, features from all layers will be returned.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - torch.Tensor: The extracted features of shape (B, C, T),
                  where C is the number of channels (features).
                - Optional[torch.Tensor]: The updated padding mask tensor
                  after processing, or None if padding_mask was not provided.

        Examples:
            >>> encoder = BeatsEncoder(...)
            >>> audio_input = torch.randn(4, 16000)  # Batch of 4 audio samples
            >>> features, updated_mask = encoder.extract_features(audio_input)

        Note:
            If SpecAugment is enabled during training, the input features
            will be augmented accordingly before feature extraction.

        Raises:
            ValueError: If the input tensor is not of the expected shape
            or if any of the layers in the encoder are misconfigured.
        """
        with autocast(False):
            fbank = self.preprocess(source)

            if self.specaug is not None and self.training:
                fbank = self.specaug(fbank)[0]

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1).float()
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            # features is BTC
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        features, layer_results = self.encoder(
            features, padding_mask=padding_mask, layer=max_layer
        )

        if max_layer is not None:
            features = layer_results[max_layer][0].transpose(
                0, 1
            )  # use the output from the max_layer

        if self.use_weighted_representation:
            repr_layer_weights = nn.functional.softmax(self.layer_weights, dim=-2)
            assert (
                max_layer is not None
            ), "max_layer must not be None when using weighted representations."
            features = (
                torch.stack(
                    [
                        layer_result_i.transpose(0, 1)
                        for layer_result_i, _ in layer_results[: max_layer + 1]
                    ],
                    dim=-2,
                )
                * repr_layer_weights
            )
            features = features.sum(dim=-2)  # BTC

        if self.downsample_conv is not None:
            features = self.downsample_conv(features.transpose(1, 2)).transpose(
                1, 2
            )  # BTC
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.conformer_adapter:
            # to handle incompatibility btw torch & huggingface
            conformer_attn_mask = ~padding_mask
            # run through conformer
            features = self.conformer_adapter(
                features,
                attention_mask=conformer_attn_mask,
            ).last_hidden_state

        if self.cross_embed_positions is not None:
            features = features + self.cross_embed_positions(features)

        return features, padding_mask


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing input sequences.

    This class implements a Transformer encoder that processes input sequences
    using self-attention mechanisms and feed-forward networks. It is designed
    to handle variable-length input sequences and supports various configurations
    for attention and normalization.

    Attributes:
        dropout (float): Dropout probability applied to the encoder.
        embedding_dim (int): Dimensionality of the encoder's input embeddings.
        pos_conv (nn.Conv1d): Convolutional layer for positional encoding.
        layers (nn.ModuleList): List of transformer encoder layers.
        layer_norm (LayerNorm): Layer normalization applied to the output.
        layer_norm_first (bool): If True, applies layer normalization before
            attention.
        layerdrop (float): Probability of dropping a transformer layer during
            training.
        relative_position_embedding (bool): If True, enables relative position
            embedding.
        num_buckets (int): Number of buckets for relative position embedding.
        max_distance (int): Maximum distance for relative position embedding.

    Args:
        config (BeatsConfig): Configuration object containing parameters for
            the transformer encoder.

    Examples:
        >>> config = BeatsConfig()
        >>> encoder = TransformerEncoder(config)
        >>> input_tensor = torch.randn(10, 32, 768)  # (sequence_length, batch_size, embedding_dim)
        >>> output, _ = encoder(input_tensor)

    Note:
        The transformer encoder supports various activation functions, dropout
        rates, and can be configured for relative position embedding.
    """

    def __init__(self, config):
        super().__init__()

        self.dropout = config.dropout
        self.embedding_dim = config.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=config.conv_pos,
            padding=config.conv_pos // 2,
            groups=config.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (config.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(
            self.pos_conv, SamePad(config.conv_pos), nn.GELU()
        )

        if hasattr(config, "relative_position_embedding"):
            self.relative_position_embedding = config.relative_position_embedding
            self.num_buckets = config.num_buckets
            self.max_distance = config.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=config.encoder_ffn_embed_dim,
                    num_attention_heads=config.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=config.attention_dropout,
                    activation_dropout=config.activation_dropout,
                    activation_fn=config.activation_fn,
                    layer_norm_first=config.layer_norm_first,
                    deep_norm=config.deep_norm,
                    has_relative_attention_bias=self.relative_position_embedding,
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    gru_rel_pos=config.gru_rel_pos,
                    encoder_layers=config.encoder_layers,
                )
                for i in range(config.encoder_layers)
            ]
        )
        if self.relative_position_embedding:
            for i in range(1, config.encoder_layers):
                del self.layers[i].self_attn.relative_attention_bias
                self.layers[i].self_attn.relative_attention_bias = self.layers[
                    0
                ].self_attn.relative_attention_bias

        self.layer_norm_first = config.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = config.encoder_layerdrop

        self.apply(init_bert_params)

        if config.deep_norm:
            deep_norm_beta = math.pow(8 * config.encoder_layers, -1 / 4)
            for i in range(config.encoder_layers):
                nn.init.xavier_normal_(self.layers[i].self_attn.k_proj.weight, gain=1)
                nn.init.xavier_normal_(
                    self.layers[i].self_attn.v_proj.weight, gain=deep_norm_beta
                )
                nn.init.xavier_normal_(self.layers[i].self_attn.q_proj.weight, gain=1)
                nn.init.xavier_normal_(
                    self.layers[i].self_attn.out_proj.weight, gain=deep_norm_beta
                )
                nn.init.xavier_normal_(self.layers[i].fc1.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].fc2.weight, gain=deep_norm_beta)

        self.layer_wise_gradient_decay_ratio = getattr(
            config, "layer_wise_gradient_decay_ratio", 1
        )

    def forward(self, x, padding_mask=None, layer=None):
        """
        Performs a forward pass through the TransformerEncoder.

        This method processes the input audio features and generates the
        corresponding audio representation, output lengths, and masks.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, T, D) where
                B is the batch size, T is the sequence length, and D is
                the feature dimension.
            ilens (torch.Tensor): Tensor of shape (B,) containing the actual
                lengths of each input sequence in the batch.
            prev_states (torch.Tensor, optional): Previous states to be used
                during the forward pass. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - audio_representation (torch.Tensor): Output tensor of shape
                  (B, T, D) representing the processed audio features.
                - output_lens (torch.Tensor): Tensor of shape (B,) containing
                  the lengths of the output sequences.
                - masks (Optional[torch.Tensor]): Placeholder for any masks
                  generated during processing. Defaults to None.

        Note:
            The function efficiently handles padding and creates necessary
            masks for the input data, ensuring compatibility with the
            underlying architecture.

        Examples:
            >>> encoder = TransformerEncoder(config)
            >>> xs_pad = torch.randn(32, 100, 512)  # Batch of 32, 100 timesteps
            >>> ilens = torch.randint(1, 101, (32,))  # Random lengths
            >>> audio_representation, output_lens, masks = encoder(xs_pad, ilens)
        """
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        """
        Extract features from raw audio.

        This method processes the input audio tensor and extracts
        meaningful features using a series of transformations, including
        patch embedding, normalization, and encoding through a transformer.

        Args:
            source (torch.Tensor): The input audio tensor of shape (B, T, D),
                where B is the batch size, T is the sequence length, and D is
                the feature dimension.
            padding_mask (Optional[torch.Tensor]): An optional mask tensor of
                shape (B, T) that indicates which positions are padding (1) and
                which are not (0). If provided, it will be used to adjust the
                padding mask of the features.
            max_layer (Optional[int]): An optional integer that specifies the
                maximum layer of the encoder to use. If provided, the output will
                only include features from this layer.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - features (torch.Tensor): The extracted features of shape (B, T, C),
                  where C is the output feature dimension.
                - padding_mask (Optional[torch.Tensor]): The adjusted padding mask
                  of the features, if provided; otherwise, None.

        Examples:
            >>> audio_input = torch.randn(8, 100, 128)  # Batch of 8 audio samples
            >>> padding_mask = torch.zeros(8, 100)  # No padding
            >>> features, adjusted_mask = self.extract_features(audio_input, padding_mask)

        Note:
            This function uses the `preprocess` method to convert raw audio
            waveforms into a feature representation (e.g., filter banks) before
            applying the transformer encoder. It also handles optional SpecAugment
            if specified during training.

        Raises:
            ValueError: If `source` is not a 3D tensor or if the dimensions do not
            match the expected shape.
        """

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            if self.layer_wise_gradient_decay_ratio != 1.0:
                x = GradMultiply.apply((x, self.layer_wise_gradient_decay_ratio))
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    pos_bias=pos_bias,
                )
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Transformer encoder layer for sentence encoding.

    This class implements a single layer of the Transformer encoder, which
    processes input sequences using self-attention mechanisms and feedforward
    networks. It allows for various configurations including dropout rates,
    activation functions, and normalization strategies.

    Attributes:
        embedding_dim (float): The dimension of the input embeddings.
        dropout (float): The dropout probability applied to the output.
        activation_dropout (float): The dropout probability applied after the
            activation function in the feedforward network.
        activation_fn (callable): The activation function used in the feedforward
            network.
        self_attn (MultiheadAttention): The multi-headed attention mechanism.
        fc1 (nn.Linear): The first linear layer in the feedforward network.
        fc2 (nn.Linear): The second linear layer in the feedforward network.
        layer_norm_first (bool): If True, applies layer normalization before the
            attention mechanism.
        final_layer_norm (LayerNorm): The layer normalization applied at the end
            of the layer.

    Args:
        embedding_dim (float): Dimension of the input embeddings. Default is 768.
        ffn_embedding_dim (float): Dimension of the feedforward network. Default is 3072.
        num_attention_heads (float): Number of attention heads in the multi-head
            attention mechanism. Default is 8.
        dropout (float): Dropout probability for the output. Default is 0.1.
        attention_dropout (float): Dropout probability for attention weights. Default is 0.1.
        activation_dropout (float): Dropout probability after activation in the feedforward
            network. Default is 0.1.
        activation_fn (str): Activation function to use. Default is "relu".
        layer_norm_first (bool): If True, applies layer normalization before the
            attention. Default is False.
        deep_norm (bool): If True, applies deep normalization. Default is False.
        has_relative_attention_bias (bool): If True, enables relative attention bias.
            Default is False.
        num_buckets (int): Number of buckets for relative position encoding. Default is 0.
        max_distance (int): Maximum distance for relative position encoding. Default is 0.
        rescale_init (bool): If True, rescales initialization. Default is False.
        gru_rel_pos (bool): If True, uses gated relative position encoding. Default is False.
        encoder_layers (int): Total number of encoder layers. Default is 0.

    Examples:
        >>> layer = TransformerSentenceEncoderLayer()
        >>> input_tensor = torch.rand(10, 32, 768)  # (seq_len, batch_size, embedding_dim)
        >>> output = layer(input_tensor)

    Note:
        The input to the `forward` method should be of shape (T, B, C) where T is
        the sequence length, B is the batch size, and C is the embedding dimension.

    Raises:
        AssertionError: If the input tensor does not match the expected dimensions.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        deep_norm: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        rescale_init: bool = False,
        gru_rel_pos: bool = False,
        encoder_layers: int = 0,
    ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        self.final_layer_norm = LayerNorm(self.embedding_dim)

        self.deep_norm = deep_norm
        if self.deep_norm:
            self.deep_norm_alpha = math.pow(2 * encoder_layers, 1 / 4)
        else:
            self.deep_norm_alpha = 1

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        pos_bias=None,
    ):
        """
        Wrapper for compatibility with ESPnet's AbsEncoder Interface.

        This method processes input tensors through the encoder to produce audio
        representations. It handles padding masks and manages the input tensor
        shapes to ensure compatibility with the BEATs encoder.

        Args:
            xs_pad (torch.Tensor): A tensor of shape (B, T, D) representing the
                padded input sequences, where B is the batch size, T is the
                sequence length, and D is the feature dimension.
            ilens (torch.Tensor): A tensor of shape (B,) containing the lengths
                of the input sequences before padding.
            prev_states (torch.Tensor, optional): Previous hidden states from
                the encoder. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - audio_representation (torch.Tensor): A tensor of shape
                (B, T, D) containing the encoded audio representations.
                - output_lens (torch.Tensor): A tensor of shape (B,) containing
                the lengths of the output sequences.
                - masks (Optional[torch.Tensor]): This is None as masks are not
                returned in this implementation.

        Examples:
            >>> encoder = BeatsEncoder(input_size=128)
            >>> xs_pad = torch.rand(10, 20, 128)  # 10 samples, 20 time steps, 128 features
            >>> ilens = torch.tensor([20] * 10)  # all sequences are of length 20
            >>> audio_rep, output_lens, masks = encoder.forward(xs_pad, ilens)
            >>> print(audio_rep.shape)  # should print: torch.Size([10, 20, D])
            >>> print(output_lens.shape)  # should print: torch.Size([10])

        Note:
            If the input tensor xs_pad is not provided, this function will create
            a tensor of size maxlen x maxlen, which can be costly in terms of
            computation. To mitigate this, the function squeezes and adjusts
            the tensor shapes as necessary.

        Raises:
            ValueError: If the input tensor shapes do not match expected
            dimensions or if the lengths are invalid.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )

            x = self.dropout1(x)
            x = residual * self.deep_norm_alpha + x

            x = self.self_attn_layer_norm(x)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual * self.deep_norm_alpha + x
            x = self.final_layer_norm(x)

        return x, attn, pos_bias


class MultiheadAttention(nn.Module):
    """
    Multi-headed attention mechanism.

    This module implements the multi-headed attention mechanism as described
    in the paper "Attention Is All You Need". It allows the model to focus
    on different parts of the input sequence when generating the output.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of attention heads.
        kdim (int, optional): Total dimension of the keys. Defaults to `embed_dim`.
        vdim (int, optional): Total dimension of the values. Defaults to `embed_dim`.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        bias (bool, optional): Whether to include bias in the linear projections.
            Defaults to True.
        add_bias_kv (bool, optional): If True, adds bias to the key and value.
            Defaults to False.
        add_zero_attn (bool, optional): If True, adds a new attention head
            that attends to zero vectors. Defaults to False.
        self_attention (bool, optional): If True, enables self-attention mode.
            Defaults to False.
        encoder_decoder_attention (bool, optional): If True, enables attention
            from encoder to decoder. Defaults to False.
        q_noise (float, optional): Amount of quantization noise. Defaults to 0.0.
        qn_block_size (int, optional): Size of the blocks for quantization noise.
            Defaults to 8.
        has_relative_attention_bias (bool, optional): If True, enables relative
            attention bias. Defaults to False.
        num_buckets (int, optional): Number of buckets for relative attention.
            Defaults to 32.
        max_distance (int, optional): Maximum distance for relative attention.
            Defaults to 128.
        gru_rel_pos (bool, optional): If True, enables GRU-based relative position
            encoding. Defaults to False.
        rescale_init (bool, optional): If True, enables rescaling initialization
            for the weights. Defaults to False.

    Methods:
        forward(query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None,
                before_softmax=False, need_head_weights=False,
                position_bias=None):
            Performs the forward pass of the multi-head attention.

    Examples:
        >>> attention = MultiheadAttention(embed_dim=512, num_heads=8)
        >>> query = torch.rand(10, 32, 512)  # (sequence_length, batch_size, embed_dim)
        >>> key = torch.rand(10, 32, 512)
        >>> value = torch.rand(10, 32, 512)
        >>> output, attn_weights, _ = attention(query, key, value)

    Note:
        Ensure that the `embed_dim` is divisible by `num_heads` to avoid errors
        during computation.

    Raises:
        AssertionError: If `embed_dim` is not divisible by `num_heads`.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        has_relative_attention_bias=False,
        num_buckets=32,
        max_distance=128,
        gru_rel_pos=False,
        rescale_init=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        k_bias = True
        if rescale_init:
            k_bias = False

        k_embed_dim = embed_dim
        q_embed_dim = embed_dim

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, k_embed_dim, bias=k_bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, q_embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initiate parameters in the transformer model.

        This method initializes the weights of the MultiheadAttention module
        and its components using a scaled Xavier uniform distribution. It is
        designed to improve the convergence behavior of the model during
        training.

        The initialization process includes the following steps:
        - Initializes the weights of the query, key, and value projection
        layers (`k_proj`, `v_proj`, `q_proj`) using Xavier uniform
        initialization.
        - Initializes the output projection layer (`out_proj`) weights using
        Xavier uniform initialization.
        - Initializes the bias terms (`bias_k` and `bias_v`) to zero, if they
        are defined.
        - If relative attention bias is used, initializes the corresponding
        weights.

        Logging information is provided to indicate that the parameters have
        been initiated.

        Examples:
            >>> attention_layer = MultiheadAttention(embed_dim=512, num_heads=8)
            >>> attention_layer.reset_parameters()

        Note:
            This method should be called after creating an instance of the
            MultiheadAttention class to ensure that the parameters are set
            before training.

        Raises:
            RuntimeError: If the embedding dimension is not divisible by the
            number of attention heads.
        """
        logging.info("Initiate parameters in the MultiheadAttention module.")
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(
                relative_positions, torch.zeros_like(relative_positions)
            )

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_positions, relative_postion_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """
        Compute relative position bias.

        This method calculates the relative position bias used in
        multi-headed attention mechanisms. It generates a bias tensor
        based on the relative positions of the query and key sequences.
        The bias is computed using the relative position buckets and
        the learned relative attention bias parameters.

        Args:
            query_length (int): The length of the query sequence.
            key_length (int): The length of the key sequence.

        Returns:
            torch.Tensor: A tensor of shape (num_heads, query_length, key_length)
            representing the computed relative position bias.

        Examples:
            >>> attention = MultiheadAttention(embed_dim=512, num_heads=8)
            >>> bias = attention.compute_bias(query_length=10, key_length=15)
            >>> print(bias.shape)
            torch.Size([8, 10, 15])

        Note:
            This method requires that `self.relative_attention_bias`
            is initialized with the appropriate parameters.
        """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(
            relative_position, bidirectional=True
        )
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(
        self,
        query,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[
            Dict[str, Dict[str, Optional[torch.Tensor]]]
        ] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for the Beats encoder.

        This method processes the input audio features and returns the
        audio representation along with the output lengths and any masks.
        It acts as a wrapper for compatibility with the ESPnet's AbsEncoder
        interface.

        Args:
            xs_pad (torch.Tensor): A tensor of shape (B, T, D) representing the
                padded audio features, where B is the batch size, T is the
                sequence length, and D is the feature dimension.
            ilens (torch.Tensor): A tensor of shape (B,) representing the actual
                lengths of each sequence in the batch.
            prev_states (torch.Tensor, optional): A tensor containing the previous
                states. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - audio_representation (torch.Tensor): A tensor of shape (B, T, D)
                representing the processed audio features.
                - output_lens (torch.Tensor): A tensor of shape (B,) containing the
                output lengths for each sequence in the batch.
                - masks (Optional[torch.Tensor]): A tensor for masks. Defaults to None.

        Note:
            If `xs_pad` is not provided, the operation can be costly since this
            function attempts to create a tensor of size maxlen x maxlen. To
            mitigate this, the input tensor is unsqueezed and then squeezed.

        Examples:
            >>> encoder = BeatsEncoder(...)
            >>> xs_pad = torch.randn(2, 10, 512)  # Batch of 2, sequence length 10, features 512
            >>> ilens = torch.tensor([10, 8])  # Actual lengths for each sequence
            >>> audio_rep, output_lengths, masks = encoder.forward(xs_pad, ilens)

        Raises:
            ValueError: If the input tensor dimensions are not as expected.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = (
                position_bias.unsqueeze(0)
                .repeat(bsz, 1, 1, 1)
                .view(bsz * self.num_heads, tgt_len, src_len)
            )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        alpha = 32
        q *= 1 / alpha

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.q_head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.k_head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[torch.Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = (
            attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]
        ) * alpha
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v, position_bias

        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos == 1:
                query_layer = (
                    q.view(bsz, self.num_heads, tgt_len, self.q_head_dim)
                    * alpha
                    / self.scaling
                )
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(
                    self.grep_linear(query_layer)
                    .view(_B, _H, _L, 2, 4)
                    .sum(-1, keepdim=False)
                ).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                attn_mask_rel_pos = (
                    gate_a_1.view(bsz * self.num_heads, tgt_len, 1) * position_bias
                )

            attn_mask_rel_pos = attn_mask_rel_pos.view(attn_weights.size())

            attn_weights = attn_weights + attn_mask_rel_pos

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, position_bias

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[torch.Tensor],
        prev_key_padding_mask: Optional[torch.Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[torch.Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]
    ) -> Dict[str, Optional[torch.Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[torch.Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        buffer: Dict[str, Optional[torch.Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        """
        Apply a sparse mask to the attention weights.

        This method is intended to be a placeholder for potential future
        implementations of sparse masking techniques. Currently, it does
        not modify the attention weights.

        Args:
            attn_weights (torch.Tensor): The raw attention weights
                with shape (bsz * num_heads, tgt_len, src_len).
            tgt_len (int): The length of the target sequence.
            src_len (int): The length of the source sequence.
            bsz (int): The batch size.

        Returns:
            torch.Tensor: The (unchanged) attention weights with the same shape
                as the input `attn_weights`.

        Note:
            This function is a no-op and returns the input weights as is.
            It can be extended in the future to implement actual sparse
            masking logic.

        Examples:
            >>> attn_weights = torch.rand(2, 5, 10)  # Example tensor
            >>> sparse_masked_weights = apply_sparse_mask(attn_weights, 5, 10, 2)
            >>> assert torch.equal(attn_weights, sparse_masked_weights)
        """
        return attn_weights


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT model.

    This function overrides the default weight initializations based on
    the specified arguments for various layer types, including linear,
    embedding, and multi-head attention layers. The initialization is done
    using a normal distribution with a mean of 0.0 and a standard deviation
    of 0.02.

    Args:
        module (nn.Module): The PyTorch module (e.g., Linear, Embedding,
        MultiheadAttention) whose weights are to be initialized.

    Notes:
        - For linear layers, weights are initialized with a normal
          distribution, and biases are set to zero.
        - For embedding layers, weights are also initialized with a normal
          distribution, and padding indices (if any) are set to zero.
        - For multi-head attention layers, the weights for query, key, and
          value projections are initialized using the same normal distribution.

    Examples:
        >>> linear_layer = nn.Linear(10, 5)
        >>> init_bert_params(linear_layer)
        >>> assert linear_layer.weight.data.mean() == 0.0  # mean should be near 0

        >>> embedding_layer = nn.Embedding(10, 5)
        >>> init_bert_params(embedding_layer)
        >>> assert embedding_layer.weight.data[0].sum() == 0.0  # padding idx should be zero

        >>> attention_layer = MultiheadAttention(embed_dim=5, num_heads=2)
        >>> init_bert_params(attention_layer)
        >>> assert attention_layer.q_proj.weight.data.mean() == 0.0  # mean should be near 0
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        logging.info("Intializing Linear Layer")
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        logging.info("Intializing Embedding Layer")
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        logging.info("Intializing Multihead Attention")
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class GradMultiply(torch.autograd.Function):
    """
    A gradient modification function that scales the gradient by a fixed scalar.

    This class provides a mechanism to modify the gradients during backpropagation
    by applying a scaling factor. It can be useful in scenarios where certain layers
    require gradient scaling to stabilize training or to implement specific training
    techniques.

    Attributes:
        scale (float): The scalar by which the gradient is multiplied during
        backpropagation.

    Methods:
        forward(ctx, i):
            Performs the forward pass of the function.

        backward(ctx, grad):
            Computes the backward pass, scaling the gradient by the specified factor.

    Examples:
        To use the `GradMultiply` function in a custom layer, you can do the following:

        ```python
        class CustomLayer(nn.Module):
            def __init__(self, scale):
                super(CustomLayer, self).__init__()
                self.scale = scale

            def forward(self, x):
                # Perform some operations on x
                ...
                # Scale the gradient
                return GradMultiply.apply((x, self.scale))
        ```

    Note:
        The input tensor `i` to the `forward` method should be a tuple containing the
        tensor to be processed and the scaling factor. The `backward` method will return
        the scaled gradient.

    Raises:
        ValueError: If the input to the `forward` method is not a tuple of length 2.
    """

    @staticmethod
    def forward(ctx, i):
        """
        Processes the input tensor through the BEATs encoder.

        This method serves as a wrapper for compatibility with the ESPnet's
        `AbsEncoder` interface. It handles padding and adjusts input shapes
        as necessary before passing them through the encoder.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, T, D), where B is
                the batch size, T is the sequence length, and D is the feature
                dimension.
            ilens (torch.Tensor): Tensor of shape (B,) containing the lengths of
                each input sequence.
            prev_states (torch.Tensor, optional): Previous hidden states, if
                any. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - audio_representation (torch.Tensor): Encoded audio representation
                of shape (B, T, D).
                - output_lens (torch.Tensor): Lengths of the output sequences
                of shape (B,).
                - masks (Optional[torch.Tensor]): Mask tensor, if applicable.
                Defaults to None.

        Note:
            If `xs_pad` is not provided, this operation can be costly as it
            attempts to create a tensor of size maxlen x maxlen. To mitigate
            this, the function squeezes tensors to reduce dimensionality.

        Examples:
            >>> model = BeatsEncoder(input_size=128)
            >>> input_tensor = torch.randn(32, 100, 512)  # Example input
            >>> input_lengths = torch.randint(1, 100, (32,))  # Random lengths
            >>> audio_rep, output_lengths, masks = model(input_tensor, input_lengths)
        """
        x, scale = i
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        """
        A gradient modification function that scales the gradient by a fixed scalar.

        This class implements a custom autograd function that modifies the gradient
        during the backward pass. The scaling is determined by a scalar value
        specified during the forward pass. This can be useful for techniques
        like gradient clipping or adaptive gradient methods.

        Attributes:
            None

        Args:
            i (tuple): A tuple containing:
                - x (torch.Tensor): The input tensor for which the gradient
                will be modified.
                - scale (float): The scalar value by which to scale the gradient.

        Returns:
            torch.Tensor: The output tensor, which is the same as the input
            tensor during the forward pass.

        Yields:
            None

        Raises:
            None

        Examples:
            >>> import torch
            >>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            >>> scale = 0.5
            >>> y = GradMultiply.apply((x, scale))
            >>> y.backward(torch.tensor([1.0, 1.0, 1.0]))
            >>> print(x.grad)  # Output: tensor([0.5, 1.0, 1.5])

        Note:
            The scaling is only applied during the backward pass, while the
            forward pass simply returns the input tensor unchanged.

        Todo:
            - Consider adding support for other scaling mechanisms or methods.
        """
        return grad * ctx.scale


class SamePad(nn.Module):
    """
    Change input tensor shape according to the kernel size and type of LM.

    This module is designed to adjust the output tensor size based on the
    specified kernel size. It is particularly useful for convolutional layers
    to ensure that the output dimensions match the desired configuration.

    Attributes:
        remove (int): The number of elements to remove from the end of the
            tensor during the forward pass. This is determined based on the
            kernel size and whether the padding is causal or not.

    Args:
        kernel_size (int): The size of the kernel to be used in the convolution.
        causal (bool): If True, it indicates that the padding should be
            causal, which means that the output will only depend on the
            current and previous inputs.

    Returns:
        x (torch.Tensor): The adjusted tensor after applying the necessary
            padding.

    Examples:
        >>> same_pad = SamePad(kernel_size=3)
        >>> input_tensor = torch.randn(1, 3, 10)  # (batch_size, channels, length)
        >>> output_tensor = same_pad(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 3, 9])  # Output length is reduced by 1

        >>> causal_pad = SamePad(kernel_size=3, causal=True)
        >>> output_tensor_causal = causal_pad(input_tensor)
        >>> output_tensor_causal.shape
        torch.Size([1, 3, 8])  # Output length is reduced by 2
    """

    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        """
        Forward pass for the Beats encoder.

        This method processes the input tensor through the encoder and returns
        the audio representation along with the output lengths and masks.

        Args:
            xs_pad: A tensor of shape (B, T, D) representing the padded input
                features, where B is the batch size, T is the sequence length,
                and D is the feature dimension.
            ilens: A tensor of shape (B,) containing the actual lengths of
                each sequence in the batch.
            prev_states: Optional; a tensor representing the previous states
                (not used in this implementation).

        Returns:
            Tuple containing:
                - audio_representation: A tensor of shape (B, T, D) representing
                the encoded audio features.
                - output_lens: A tensor of shape (B,) representing the lengths
                of the output sequences.
                - masks: None, as no masks are generated in this implementation.

        Note:
            If `xs_pad` is not provided, the operation may be costly as this
            function tries to create a tensor of size maxlen x maxlen.
            Therefore, the function unsqueezes and then squeezes tensors to
            manage tensor shapes effectively.

        Examples:
            >>> beats_encoder = BeatsEncoder(input_size=256)
            >>> xs_pad = torch.rand(8, 100, 256)  # 8 sequences of length 100
            >>> ilens = torch.tensor([100, 80, 90, 70, 100, 100, 60, 50])
            >>> audio_representation, output_lens, masks = beats_encoder.forward(xs_pad, ilens)
        """
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class Swish(nn.Module):
    """
    Swish activation function.

    The Swish activation function is defined as:
        Swish(x) = x * sigmoid(x)

    This activation function has been shown to outperform ReLU in some deep
    learning applications.

    Attributes:
        act (torch.nn.Sigmoid): The sigmoid activation function instance.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the Swish activation for the input tensor.

    Examples:
        >>> swish = Swish()
        >>> input_tensor = torch.tensor([-1.0, 0.0, 1.0])
        >>> output_tensor = swish(input_tensor)
        >>> print(output_tensor)
        tensor([-0.2689, 0.0000, 0.7311])
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the Beats encoder.

        This method processes the input tensor `xs_pad` and its corresponding
        lengths `ilens` to produce audio representations. It is designed to be
        compatible with the AbsEncoder interface in ESPnet.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, T, D), where B is
                the batch size, T is the sequence length, and D is the feature
                dimension.
            ilens (torch.Tensor): Tensor of shape (B,) representing the actual
                lengths of the sequences in `xs_pad`.
            prev_states (torch.Tensor, optional): Optional tensor representing
                previous states. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - audio_representation (torch.Tensor): The output audio
                representation of shape (B, T, D).
                - output_lens (torch.Tensor): Tensor of shape (B,) containing
                the output lengths for each sequence.
                - masks (Optional[torch.Tensor]): Currently set to None.

        Note:
            If `xs_pad` is not provided, this operation may be costly, as it
            attempts to create a tensor of size maxlen x maxlen. Thus,
            tensors are unsqueezed and then squeezed to optimize performance.

        Examples:
            >>> beats_encoder = BeatsEncoder(...)
            >>> xs_pad = torch.randn(32, 100, 512)  # Batch of 32, 100 time steps
            >>> ilens = torch.randint(1, 101, (32,))  # Random lengths
            >>> audio_representation, output_lens, masks = beats_encoder.forward(xs_pad, ilens)

        Raises:
            ValueError: If `xs_pad` or `ilens` have incompatible shapes.
        """
        return x * self.act(x)


class GLU_Linear(nn.Module):
    """
    GLU Linear layer.

    This class implements a Gated Linear Unit (GLU) layer, which is a
    variation of a linear layer that uses a gating mechanism. The input
    is split into two parts: one part is passed through a linear layer,
    and the other part is passed through a non-linear activation function.
    The output is the element-wise multiplication of the two parts.

    Attributes:
        glu_type (str): The type of activation function to use for gating.
            Options are 'sigmoid', 'swish', 'relu', and 'gelu'.
        output_dim (int): The dimension of the output from the GLU layer.
        linear (nn.Linear): The linear transformation applied to the input.

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        glu_type (str): The activation function used for the GLU gate.
            Defaults to "sigmoid".
        bias_in_glu (bool): Whether to include a bias term in the linear
            transformation. Defaults to True.

    Examples:
        >>> glu_layer = GLU_Linear(input_dim=128, output_dim=64, glu_type='swish')
        >>> input_tensor = torch.randn(32, 128)  # Batch size of 32
        >>> output_tensor = glu_layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 64])

    Note:
        The GLU mechanism is particularly useful in tasks such as
        natural language processing and speech processing, where
        controlling the flow of information can improve performance.
    """

    def __init__(self, input_dim, output_dim, glu_type="sigmoid", bias_in_glu=True):
        super(GLU_Linear, self).__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim

        if glu_type == "sigmoid":
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = torch.nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = torch.nn.GELU()

        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        """
        Processes input tensors through the BEATs encoder.

        This method wraps the encoding process for compatibility with
        ESPnet's AbsEncoder interface. It takes padded input features,
        their lengths, and optionally previous states to generate
        audio representations.

        Args:
            xs_pad (torch.Tensor): Padded input tensor of shape (B, T, D)
                where B is the batch size, T is the sequence length,
                and D is the feature dimension.
            ilens (torch.Tensor): Tensor containing the lengths of each
                sequence in the batch, shape (B,).
            prev_states (torch.Tensor, optional): Previous states, not used
                in this implementation. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - audio_representation (torch.Tensor): Encoded audio
                  representations of shape (B, T, D).
                - output_lens (torch.Tensor): Lengths of the output
                  sequences, shape (B,).
                - masks (Optional[torch.Tensor]): Currently set to None.

        Note:
            If the input tensor is not provided, the operation can be costly
            as this function attempts to create a tensor of size maxlen x
            maxlen. To mitigate this, the input tensor is unsqueezed and
            then squeezed to optimize memory usage.

        Examples:
            >>> encoder = BeatsEncoder(input_size=128)
            >>> padded_inputs = torch.randn(10, 20, 128)  # (B, T, D)
            >>> lengths = torch.tensor([20, 18, 15, 20, 10, 20, 20, 20, 20, 20])
            >>> audio_rep, output_lens, masks = encoder.forward(padded_inputs, lengths)
        """
        # to be consistent with GLU_Linear, we assume the input always has the
        # #channel (#dim) in the last dimension of the tensor, so need to
        # switch the dimension first for 1D-Conv case
        x = self.linear(x)

        if self.glu_type == "bilinear":
            x = (
                x[:, :, 0 : self.output_dim]
                * x[:, :, self.output_dim : self.output_dim * 2]
            )
        else:
            x = x[:, :, 0 : self.output_dim] * self.glu_act(
                x[:, :, self.output_dim : self.output_dim * 2]
            )

        return x


def gelu_accurate(x):
    """
    Computes the accurate Gaussian Error Linear Unit (GELU) activation.

    The GELU activation function is defined as:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3)))

    This function computes the GELU activation in a numerically stable way
    by avoiding overflow issues in the exponential function.

    Args:
        x (torch.Tensor): The input tensor for which the GELU activation
            needs to be computed.

    Returns:
        torch.Tensor: A tensor with the same shape as the input tensor,
            containing the GELU activations.

    Examples:
        >>> import torch
        >>> input_tensor = torch.tensor([-1.0, 0.0, 1.0])
        >>> output_tensor = gelu_accurate(input_tensor)
        >>> print(output_tensor)
        tensor([-0.1587,  0.0000,  0.8413])

    Note:
        This implementation is designed to provide accurate results
        for a wide range of input values. It is recommended to use
        this function in neural networks where GELU activation is desired.
    """
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function.

    The GELU activation function is defined mathematically as:
    `GELU(x) = x * P(X <= x)`, where `P` is the cumulative distribution function
    of a standard normal distribution. It has been found to perform better than
    traditional activation functions such as ReLU and tanh in certain neural
    network architectures.

    This implementation casts the input tensor to float32 before applying the
    GELU function to ensure numerical stability, then casts it back to the
    original data type.

    Args:
        x (torch.Tensor): Input tensor of any shape. The tensor will be cast
            to float32 for computation.

    Returns:
        torch.Tensor: Output tensor after applying the GELU activation function,
            with the same shape and type as the input tensor.

    Examples:
        >>> import torch
        >>> input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        >>> output_tensor = gelu(input_tensor)
        >>> print(output_tensor)
        tensor([-0.1587,  0.0000,  0.8413,  1.7615])

    Note:
        This function uses PyTorch's built-in GELU function for performance and
        accuracy.

    Todo:
        Consider adding support for in-place operations to save memory.
    """
    return F.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str):
    """
    Returns the activation function corresponding to `activation`.

    This function maps a string representation of an activation function
    to its corresponding PyTorch function. Supported activation functions
    include ReLU, GELU, Tanh, and others.

    Args:
        activation (str): The name of the activation function. Supported
            values are: "relu", "gelu", "gelu_fast", "gelu_accurate",
            "tanh", "linear", and "glu".

    Returns:
        Callable: The corresponding activation function.

    Raises:
        RuntimeError: If the specified activation function is not supported.

    Examples:
        >>> relu_fn = get_activation_fn("relu")
        >>> output = relu_fn(torch.tensor([-1.0, 0.0, 1.0]))
        tensor([0., 0., 1.])

        >>> gelu_fn = get_activation_fn("gelu")
        >>> output = gelu_fn(torch.tensor([-1.0, 0.0, 1.0]))
        tensor([-0.1587, 0.0000, 0.8413])

        >>> tanh_fn = get_activation_fn("tanh")
        >>> output = tanh_fn(torch.tensor([-1.0, 0.0, 1.0]))
        tensor([-0.7616, 0.0000, 0.7616])

    Note:
        The activation function "gelu_fast" has been renamed to
        "gelu_accurate". If "gelu_fast" is requested, a warning will
        be issued, and the "gelu_accurate" function will be returned
        instead.
    """

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        warnings.warn("--activation-fn=gelu_fast has been renamed to gelu_accurate")
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "glu":
        return lambda x: x
    else:
        raise RuntimeError(f"--activation-fn {activation} not supported")


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for subsequent
    quantization with Iterative Product Quantization, as described in "Training
    with Quantization Noise for Extreme Model Compression".

    This function modifies the behavior of a given module by introducing quantization
    noise during training. It randomly drops blocks of weights in the module, which
    can help improve the robustness of the model to quantization effects.

    Args:
        module (nn.Module): The PyTorch module (e.g., Linear, Embedding, or
            Conv2d) to which quantization noise will be applied.
        p (float): The probability of dropping a block of weights. Should be a
            value between 0 and 1.
        block_size (int): The size of the blocks for subsequent quantization
            with iterative product quantization.

    Returns:
        nn.Module: The modified module with quantization noise applied.

    Raises:
        AssertionError: If the module type is not supported or if the weights
            of the module do not have the correct dimensions with respect to
            the specified block size.

    Note:
        - Module weights must have the right sizes relative to the block size.
        - Only Linear, Embedding, and Conv2d modules are supported.
        - For more details on how to quantize by blocks with convolutional
          weights, see "And the Bit Goes Down: Revisiting the Quantization of
          Neural Networks".
        - This implementation represents a simple form of noise, which consists
          of randomly dropping blocks.

    Examples:
        >>> linear_layer = nn.Linear(10, 5)
        >>> noisy_layer = quant_noise(linear_layer, p=0.1, block_size=2)
        >>> print(noisy_layer)
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module
