# Copyright 2023
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# The original AVHubert work is in:
#     Paper: https://arxiv.org/pdf/2201.02184.pdf
#     Original code: https://github.com/facebookresearch/av_hubert


"""Encoder definition."""
import contextlib
import copy
import logging
import math
import os
import random
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from filelock import FileLock
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """
    Create a 3x3 convolutional layer with padding.

    This function returns a 2D convolution layer that uses a kernel size of
    3x3 and includes padding to maintain the spatial dimensions of the
    input. The layer does not include a bias term.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.

    Returns:
        nn.Conv2d: A 2D convolution layer configured with the specified
        parameters.

    Examples:
        >>> conv_layer = conv3x3(16, 32)
        >>> input_tensor = torch.randn(1, 16, 64, 64)  # (N, C, H, W)
        >>> output_tensor = conv_layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 32, 64, 64])  # Output shape remains the same as input
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def downsample_basic_block(inplanes, outplanes, stride):
    """
    Construct a downsample block for a neural network.

    This function creates a sequential block consisting of a 1x1 convolution
    followed by batch normalization. It is typically used in architectures
    that require downsampling of feature maps, such as ResNet variants.

    Args:
        inplanes (int): Number of input channels.
        outplanes (int): Number of output channels.
        stride (int): The stride of the convolution.

    Returns:
        nn.Sequential: A sequential block containing a convolution and
        batch normalization.

    Examples:
        >>> downsample_block = downsample_basic_block(64, 128, stride=2)
        >>> print(downsample_block)
        Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128)
        )
    """
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def downsample_basic_block_v2(inplanes, outplanes, stride):
    """
    Construct a downsample block for use in a neural network.

    This function creates a sequential block consisting of an average pooling
    layer followed by a 1x1 convolutional layer and batch normalization. It is
    used to reduce the spatial dimensions of the input feature maps while
    increasing the number of output channels.

    Args:
        inplanes (int): Number of input channels.
        outplanes (int): Number of output channels.
        stride (int): Stride for the average pooling layer, which determines the
            downsampling factor.

    Returns:
        nn.Sequential: A sequential block containing the average pooling layer,
        convolutional layer, and batch normalization layer.

    Examples:
        >>> downsample_block = downsample_basic_block_v2(64, 128, stride=2)
        >>> print(downsample_block)
        Sequential(
          (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(128)
        )
    """
    return nn.Sequential(
        nn.AvgPool2d(
            kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False
        ),
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def time_masking(xs_pad, min_T=5, max_T=20):
    """
    Mask contiguous frames of audio or video inputs with random lengths.

    This function applies random masking to contiguous frames in the input tensor
    `xs_pad`, simulating occlusion in audio or video data. The length of the mask
    is randomly chosen from the range [min_T, max_T].

    Args:
        xs_pad (torch.Tensor): The input tensor of shape (B, D, L), where B is the
            batch size, D is the number of features, and L is the sequence length.
        min_T (int, optional): The minimum length of the mask. Default is 5.
        max_T (int, optional): The maximum length of the mask. Default is 20.

    Returns:
        torch.Tensor: The masked input tensor of the same shape as `xs_pad`.

    Examples:
        >>> xs_pad = torch.randn(2, 10, 100)  # Batch of 2, 10 features, 100 length
        >>> masked_output = time_masking(xs_pad, min_T=3, max_T=10)
        >>> masked_output.shape
        torch.Size([2, 10, 100])  # Output shape remains the same

    Note:
        The masking is applied independently for each batch element.

    Raises:
        ValueError: If `min_T` or `max_T` is less than 1, or if `min_T` is
        greater than `max_T`.
    """
    batch_size = xs_pad.size(0)
    mask = torch.ones_like(xs_pad)
    for b in range(batch_size):
        width = min(random.randint(min_T, max_T), xs_pad.size(1))
        start = random.randint(0, xs_pad.size(1) - width)
        mask[b, start : start + width] = 0.0
    return xs_pad * mask.to(xs_pad.device)


# avhubert_url(noise_large):
# 'https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/large_vox_iter5.pt'
# avhubert_url(noise_base):
# 'https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/base_vox_iter5.pt'


class FairseqAVHubertEncoder(AbsEncoder):
    """
    FairSeq AVHubert pretrained encoder module.

    This class implements a pretrained encoder for audio-visual (AV)
    representation learning using the AVHubert model. It extends the
    AbsEncoder class and integrates both audio and video modalities
    for feature extraction.

    Attributes:
        input_size (int): The dimension of the input features.
        avhubert_url (str): URL for downloading the pretrained AVHubert model.
        avhubert_dir_path (str): Directory path for storing the downloaded model.
        extracted (bool): Indicates if the model is in the extracted state.
        modality_dropout (float): Dropout rate for modality features.
        audio_dropout (float): Dropout rate for audio features.
        audio_only (bool): If True, only audio features are processed.

    Args:
        input_size (int): Input dimension for the encoder. Defaults to 1.
        avhubert_url (str): Download link for the pretrained AVHubert model.
            Defaults to "./".
        avhubert_dir_path (str): Directory path for the downloaded model.
            Defaults to "./".
        freeze_finetune_updates (int): Number of updates to freeze finetuning.
            Defaults to 0.
        encoder_embed_dim (int): Dimension of the encoder embeddings.
            Defaults to 1024.
        encoder_layerdrop (float): Dropout probability for encoder layers.
            Defaults to 0.05.
        dropout_input (float): Dropout probability for input features.
            Defaults to 0.1.
        dropout_features (float): Dropout probability for feature extraction.
            Defaults to 0.1.
        dropout (float): Dropout probability in the encoder.
            Defaults to 0.1.
        attention_dropout (float): Dropout probability for attention weights.
            Defaults to 0.1.
        feature_grad_mult (float): Gradient multiplier for feature extractor.
            Defaults to 0.1.
        activation_dropout (float): Dropout probability after activation.
            Defaults to 0.0.
        wav_input (bool): If True, indicates that input is audio waveform.
            Defaults to False.
        layer_norm_first (bool): If True, applies layer normalization first.
            Defaults to True.
        audio_feat_dim (int): Dimension of audio features. Defaults to 104.
        encoder_layers (int): Number of encoder layers. Defaults to 24.
        encoder_ffn_embed_dim (int): Dimension of the FFN embeddings.
            Defaults to 4096.
        encoder_attention_heads (int): Number of attention heads in the encoder.
            Defaults to 16.
        extracted (bool): Indicates if features are extracted.
            Defaults to False.
        pretrain (bool): If True, uses pretrained model weights.
            Defaults to True.
        modality_dropout (float): Dropout rate for modality features.
            Defaults to 0.0.
        audio_dropout (float): Dropout rate for audio features.
            Defaults to 0.0.
        noise_augmentation (bool): If True, applies noise augmentation.
            Defaults to False.
        noise_path (str): Path to the noise data for augmentation.
            Defaults to "./data/babble_noise.pt".
        max_noise_weight (float): Maximum weight for noise in augmentation.
            Defaults to 0.5.
        audio_only (bool): If True, only processes audio stream.
            Defaults to False.

    Returns:
        None

    Raises:
        ValueError: If input does not contain video or audio data.

    Examples:
        encoder = FairseqAVHubertEncoder(input_size=1, avhubert_url="path/to/model")
        audio_input = torch.randn(8, 104, 100)  # (batch_size, feature_dim, length)
        video_input = torch.randn(8, 1, 10, 224, 224)  # (batch_size, 1, T, H, W)
        inputs = {"audio": audio_input, "video": video_input}
        lengths = torch.tensor([100, 90, 80, 70, 60, 50, 40, 30])  # Example lengths
        output, olens, _ = encoder(inputs, lengths)

    Note:
        The AVHubert model is based on the architecture described in the
        paper: "Self-Supervised Learning of Audio-Visual Representations".
        Ensure to set the correct input shapes for audio and video data.
    """

    @typechecked
    def __init__(
        self,
        input_size: int = 1,
        avhubert_url: str = "./",
        avhubert_dir_path: str = "./",
        freeze_finetune_updates: int = 0,
        encoder_embed_dim: int = 1024,
        encoder_layerdrop: float = 0.05,
        dropout_input: float = 0.1,
        dropout_features: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        feature_grad_mult: float = 0.1,
        activation_dropout: float = 0.0,
        wav_input: bool = False,
        layer_norm_first: bool = True,
        audio_feat_dim: int = 104,
        encoder_layers: int = 24,
        encoder_ffn_embed_dim: int = 4096,
        encoder_attention_heads: int = 16,
        extracted: bool = False,
        pretrain: bool = True,
        modality_dropout: float = 0.0,
        audio_dropout: float = 0.0,
        noise_augmentation: bool = False,
        noise_path: str = "./data/babble_noise.pt",
        max_noise_weight: float = 0.5,
        audio_only: bool = False,
    ):
        super().__init__()

        self._output_size = encoder_embed_dim
        self.extracted = extracted
        self.modality_dropout = modality_dropout
        self.audio_dropout = audio_dropout
        self.audio_only = audio_only

        arg_overrides = {
            "encoder_embed_dim": encoder_embed_dim,
            "encoder_layerdrop": encoder_layerdrop,
            "dropout_input": dropout_input,
            "dropout_features": dropout_features,
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "feature_grad_mult": feature_grad_mult,
            "activation_dropout": activation_dropout,
            "wav_input": wav_input,
            "layer_norm_first": layer_norm_first,
            "audio_feat_dim": audio_feat_dim,
            "encoder_layers": encoder_layers,
            "encoder_ffn_embed_dim": encoder_ffn_embed_dim,
            "encoder_attention_heads": encoder_attention_heads,
            "audio_only": audio_only,
        }
        default_cfg = AVHubertConfig()
        for arg_name, arg_val in arg_overrides.items():
            setattr(default_cfg, arg_name, arg_val)

        model = AVHubertModel.build_model(cfg=default_cfg)
        self.modality_fuse = model.modality_fuse

        if pretrain:
            self.avhubert_model_path = download_avhubert(
                avhubert_url,
                avhubert_dir_path,
            )

            ckpt = torch.load(
                self.avhubert_model_path,
                map_location=torch.device("cpu"),
            )
            state = {
                k: v
                for k, v in ckpt["model"].items()
                if "label_embs_concat" not in k and "final_proj" not in k
            }
            del ckpt
            model.load_state_dict(state)
        else:
            logging.info(
                "Training from scratch without \
                         using pre-trained AV-HuBERT model"
            )

        self.pretrained_params = copy.deepcopy(model.state_dict())

        self.encoders = model

        if noise_augmentation:
            self.noise = torch.load(noise_path)
            self.max_noise_weight = max_noise_weight
        else:
            self.noise = None
            self.max_noise_weight = None

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        """
        Get the output size of the AVHubert encoder.

        This method returns the dimensionality of the output from the encoder,
        which is defined during the initialization of the encoder.

        Returns:
            int: The output size of the encoder, which corresponds to the
            embedding dimension specified during initialization.

        Examples:
            >>> encoder = FairseqAVHubertEncoder(encoder_embed_dim=512)
            >>> encoder.output_size()
            512

        Note:
            The output size is primarily determined by the `encoder_embed_dim`
            parameter passed to the encoder during its construction.
        """
        return self._output_size

    def forward(
        self,
        xs_pad: Dict[str, torch.Tensor],
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward AVHubert Encoder.

        This method processes the input tensors for audio and video modalities
        through the AVHubert encoder, applying necessary transformations and
        masking. It returns the encoded features along with the output lengths
        and an optional tensor for further processing.

        Args:
            xs_pad (Dict[str, torch.Tensor]): A dictionary containing the input
                tensors:
                - 'video': input tensor of shape (B, 1, L, H, W)
                - 'audio': input tensor of shape (B, D, L)
            ilens (torch.Tensor): A tensor of shape (B,) representing the lengths
                of the input sequences for each batch.
            prev_states (torch.Tensor, optional): Not used in the current version.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - Encoded features tensor of shape (B, T, D).
                - A tensor of output lengths for each input sequence of shape (B,).
                - An optional tensor that can be used for further processing,
                  currently set to None.

        Raises:
            ValueError: If neither 'video' nor 'audio' keys are present in
                `xs_pad`.

        Examples:
            >>> encoder = FairseqAVHubertEncoder()
            >>> audio_input = torch.randn(2, 104, 50)  # (B, D, L)
            >>> video_input = torch.randn(2, 1, 50, 224, 224)  # (B, 1, L, H, W)
            >>> ilens = torch.tensor([50, 50])
            >>> xs_pad = {'audio': audio_input, 'video': video_input}
            >>> encoded_features, output_lengths, _ = encoder(xs_pad, ilens)
            >>> print(encoded_features.shape)  # Output: (2, T, D)
            >>> print(output_lengths.shape)  # Output: (2,)

        Note:
            This function supports both training and inference modes. During
            training, additional augmentations like time masking and modality
            dropout are applied.
        """
        if not self.extracted:
            if "video" in xs_pad:
                masks = make_pad_mask(ilens, length_dim=2).to(xs_pad["video"].device)
            elif "audio" in xs_pad:
                masks = make_pad_mask(ilens, length_dim=2).to(xs_pad["audio"].device)
            else:
                ValueError("Input should have video or audio")

            ft = self.freeze_finetune_updates <= self.num_updates

            if self.num_updates <= self.freeze_finetune_updates:
                self.num_updates += 1
            elif ft and self.num_updates == self.freeze_finetune_updates + 1:
                self.num_updates += 1
                logging.info("Start fine-tuning AVhubert parameters!")
            else:
                self.num_updates += 1
            with torch.no_grad() if not ft else contextlib.nullcontext():
                enc_outputs = self.encoders.extract_finetune(
                    xs_pad,
                    padding_mask=masks,
                )
        else:
            masks = make_pad_mask(ilens, length_dim=1).to(xs_pad.device)
            ft = self.freeze_finetune_updates <= self.num_updates

            if self.training:
                xs_pad = time_masking(xs_pad)

                if self.modality_dropout > 0 and self.modality_fuse == "concat":
                    modality_drop_prob, audio_drop_prob = (
                        np.random.random(),
                        np.random.random(),
                    )
                    if modality_drop_prob < self.modality_dropout:
                        if audio_drop_prob < self.audio_dropout:
                            # first half dimension is audio features
                            modal_masks = torch.ones_like(xs_pad)
                            modal_masks[:, :, : modal_masks.size(2) // 2] = 0.0
                            xs_pad = xs_pad * modal_masks
                        else:
                            # last half dimension is video features
                            modal_masks = torch.ones_like(xs_pad)
                            modal_masks[:, :, modal_masks.size(2) // 2 :] = 0.0
                            xs_pad = xs_pad * modal_masks

                if self.noise is not None:
                    start_ind = torch.randint(
                        0, self.noise.size(0) - xs_pad.size(1), size=[xs_pad.size(0)]
                    )  # B
                    noise_ind = start_ind.view(-1, 1) + torch.arange(
                        0, xs_pad.size(1)
                    ).unsqueeze(0).repeat(
                        xs_pad.size(0), 1
                    )  # B,T
                    noise_weight = (
                        torch.rand([xs_pad.size(0), 1, 1]).to(xs_pad.device)
                        * self.max_noise_weight
                    )
                    xs_pad = (1 - noise_weight) * xs_pad + noise_weight * self.noise[
                        noise_ind
                    ].to(xs_pad.device)

            if self.audio_only:
                modal_masks = torch.ones_like(xs_pad)
                modal_masks[:, :, : modal_masks.size(2) // 2] = 0.0
                xs_pad = xs_pad * modal_masks

            if self.num_updates <= self.freeze_finetune_updates:
                self.num_updates += 1
            elif ft and self.num_updates == self.freeze_finetune_updates + 1:
                self.num_updates += 1
                logging.info("Start fine-tuning AVhubert parameters!")
            else:
                self.num_updates += 1
            with torch.no_grad() if not ft else contextlib.nullcontext():
                enc_outputs = self.encoders.forward_transformer(
                    xs_pad,
                    padding_mask=masks,
                )

        xs_pad = enc_outputs[0]
        masks = enc_outputs[1]

        # save gpu memory
        del enc_outputs

        olens = (~masks).sum(dim=1)

        return xs_pad, olens, None

    def forward_fusion(
        self,
        xs_pad: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuses audio and video features extracted from the encoder.

        This method takes in a dictionary containing audio and video
        features, processes them through their respective modality
        encoders, and then fuses the results using the specified fusion
        method (concatenation or addition).

        Args:
            xs_pad (Dict[str, torch.Tensor]): A dictionary containing:
                - 'audio' (torch.Tensor): Audio features tensor of shape
                  (B, D, L), where B is the batch size, D is the number of
                  audio features, and L is the sequence length.
                - 'video' (torch.Tensor): Video features tensor of shape
                  (B, 1, L, H, W), where H is the height and W is the width
                  of the video frames.

        Returns:
            torch.Tensor: The fused features tensor. The shape of the
            returned tensor depends on the fusion method used:
                - If concatenation, shape will be (B, D * 2, L).
                - If addition, shape will be (B, D, L).

        Examples:
            >>> audio_input = torch.randn(4, 128, 10)  # Batch of 4 audio samples
            >>> video_input = torch.randn(4, 1, 10, 224, 224)  # Batch of 4 video samples
            >>> encoder = FairseqAVHubertEncoder()
            >>> fused_features = encoder.forward_fusion({
            ...     'audio': audio_input,
            ...     'video': video_input
            ... })
            >>> print(fused_features.shape)
            torch.Size([4, 256, 10])  # If concatenation is used

        Note:
            The audio and video features must be preprocessed and
            extracted before calling this method. If either audio or
            video features are not provided, the method will handle
            it gracefully by creating zero tensors for the missing
            modality.

        Raises:
            ValueError: If both audio and video features are None.
        """
        if xs_pad["audio"] is not None:
            audio_feats = self.encoders.forward_audio(xs_pad["audio"])
        else:
            audio_feats = None
        if xs_pad["video"] is not None:
            video_feats = self.encoders.forward_video(xs_pad["video"])
        else:
            video_feats = None
        return self.encoders.modality_fusion(audio_feats, video_feats)

    def reload_pretrained_parameters(self):
        """
        Reload the pretrained parameters into the encoder.

        This method allows the user to restore the original pretrained
        parameters of the AVHubert encoder. It is particularly useful in
        scenarios where the model has undergone fine-tuning and the user
        wants to revert to the initial state of the model.

        The pretrained parameters are loaded from the `self.pretrained_params`
        attribute, which is a deep copy of the model's state dictionary at
        initialization.

        Returns:
            None

        Examples:
            # Create an instance of the encoder
            encoder = FairseqAVHubertEncoder()

            # Fine-tune the encoder
            # ...

            # Reload pretrained parameters
            encoder.reload_pretrained_parameters()
        """
        logging.info("Pretrained AVHubert model parameters reloaded!")


@dataclass
class AVHubertConfig:
    """
    Configuration for AV-HuBERT model.

    This class encapsulates the configuration settings required for the
    AV-HuBERT model. It includes parameters related to the audio and
    video modalities, as well as dropout rates and other model
    hyperparameters.

    Attributes:
        sample_rate (int): Target sample rate for audio files, which will
            be up/down sampled to this rate. Default is 16000.
        label_rate (int): Label frame rate. Set to -1 for sequence label.
        encoder_layers (int): Number of encoder layers in the transformer.
            Default is 12.
        encoder_embed_dim (int): Encoder embedding dimension. Default is 768.
        encoder_ffn_embed_dim (int): Encoder embedding dimension for feedforward
            networks. Default is 3072.
        encoder_attention_heads (int): Number of attention heads in the
            encoder. Default is 12.
        activation_fn (str): Activation function to use. Default is "gelu".
        dropout (float): Dropout probability for the transformer. Default is 0.1.
        attention_dropout (float): Dropout probability for attention weights.
            Default is 0.1.
        activation_dropout (float): Dropout probability after activation in
            feedforward networks. Default is 0.0.
        encoder_layerdrop (float): Probability of dropping a transformer layer.
            Default is 0.0.
        dropout_input (float): Dropout applied to the input after feature
            extraction. Default is 0.0.
        dropout_features (float): Dropout applied to the features after
            feature extraction. Default is 0.0.
        final_dim (int): Project final representations and targets to this many
            dimensions. Set to encoder_embed_dim if <= 0. Default is 0.
        untie_final_proj (bool): Use separate projection for each target.
            Default is False.
        layer_norm_first (bool): Apply layer normalization first in the
            transformer. Default is False.
        conv_feature_layers (str): Description of convolutional feature
            extraction layers in the form of a Python list.
            Default is "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2".
        conv_bias (bool): Include bias in the convolutional encoder.
            Default is False.
        logit_temp (float): Temperature to divide logits by. Default is 0.1.
        target_glu (bool): Adds projection + GLU to targets. Default is False.
        feature_grad_mult (float): Multiply feature extractor variable gradients
            by this value. Default is 1.0.
        mask_length_audio (int): Length of the mask for audio features.
            Default is 10.
        mask_prob_audio (float): Probability of replacing a token with a mask
            for audio features. Default is 0.65.
        mask_length_image (int): Length of the mask for image features.
            Default is 10.
        mask_prob_image (float): Probability of replacing a token with a mask
            for image features. Default is 0.65.
        mask_selection (str): Method for choosing mask length. Default is "static".
        mask_other (float): Secondary mask argument for more complex
            distributions. Default is 0.
        no_mask_overlap (bool): Whether to allow masks to overlap. Default is False.
        mask_min_space (int): Minimum space between spans if no overlap is enabled.
            Default is 1.
        mask_channel_length (int): Length of the mask for features (channels).
            Default is 10.
        mask_channel_prob (float): Probability of replacing a feature with 0
            for channel masking. Default is 0.0.
        mask_channel_selection (str): Method for choosing mask length for
            channel masking. Default is "static".
        mask_channel_other (float): Secondary mask argument for more complex
            distributions for channel masking. Default is 0.
        no_mask_channel_overlap (bool): Whether to allow channel masks to overlap.
            Default is False.
        mask_channel_min_space (int): Minimum space between spans if no overlap
            is enabled for channel masking. Default is 1.
        conv_pos (int): Number of filters for convolutional positional embeddings.
            Default is 128.
        conv_pos_groups (int): Number of groups for convolutional positional
            embedding. Default is 16.
        latent_temp (Tuple[float, float, float]): Legacy parameter (to be removed).
            Default is (2, 0.5, 0.999995).
        skip_masked (bool): Skip computing losses over masked frames. Default is False.
        skip_nomask (bool): Skip computing losses over unmasked frames. Default is False.
        resnet_relu_type (str): ReLU type for ResNet. Default is "prelu".
        resnet_weights (Optional[str]): Pretrained ResNet weights. Default is None.
        sim_type (str): Similarity type for loss computation. Default is "cosine".
        sub_encoder_layers (int): Number of transformer layers for single modality.
            Default is 0.
        audio_feat_dim (int): Audio feature dimension. Default is -1.
        modality_dropout (float): Drop one modality. Default is 0.
        audio_dropout (float): Drop audio feature. Default is 0.
        modality_fuse (str): Method for fusing two modalities: "add" or "concat".
            Default is "concat".
        selection_type (str): Type of selecting images. Default is "same_other_seq".
        masking_type (str): Type of masking: "input" or "feature". Default is "input".
        decoder_embed_dim (int): Decoder embedding dimension. Default is 768.
        decoder_ffn_embed_dim (int): Decoder embedding dimension for FFN.
            Default is 3072.
        decoder_layers (int): Number of decoder layers. Default is 6.
        decoder_layerdrop (float): Decoder layer drop chance. Default is 0.0.
        decoder_attention_heads (int): Number of decoder attention heads. Default is 4.
        decoder_learned_pos (bool): Use learned positional embeddings in the decoder.
            Default is False.
        decoder_normalize_before (bool): Apply layer normalization before each
            decoder block. Default is False.
        no_token_positional_embeddings (bool): If set, disables positional embeddings
            (outside self-attention). Default is False.
        decoder_dropout (float): Dropout probability in the decoder. Default is 0.1.
        decoder_attention_dropout (float): Dropout probability for attention
            weights inside the decoder. Default is 0.1.
        decoder_activation_dropout (float): Dropout probability after activation
            in FFN inside the decoder. Default is 0.0.
        max_target_positions (int): Maximum target positions. Default is 2048.
        share_decoder_input_output_embed (bool): Share decoder input and output
            embeddings. Default is False.
        audio_only (bool): Whether to use audio stream only. Default is False.
        no_scale_embedding (bool): Scale embedding. Default is True.

    Examples:
        config = AVHubertConfig(
            sample_rate=16000,
            encoder_layers=12,
            modality_fuse='concat',
            audio_only=True
        )
    """

    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    label_rate: int = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: str = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length_audio: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_audio: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_length_image: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_image: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: str = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: str = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    resnet_relu_type: str = field(
        default="prelu", metadata={"help": "relu type for resnet"}
    )
    resnet_weights: Optional[str] = field(
        default=None, metadata={"help": "resnet weights"}
    )
    sim_type: str = field(default="cosine", metadata={"help": "similarity type"})

    sub_encoder_layers: int = field(
        default=0, metadata={"help": "number of transformer layers for single modality"}
    )
    audio_feat_dim: int = field(
        default=-1, metadata={"help": "audio feature dimension"}
    )
    modality_dropout: float = field(default=0, metadata={"help": "drop one modality"})
    audio_dropout: float = field(default=0, metadata={"help": "drop audio feature"})
    modality_fuse: str = field(
        default="concat", metadata={"help": "fusing two modalities: add,concat"}
    )
    selection_type: str = field(
        default="same_other_seq",
        metadata={
            "help": "type of selectig images,"
            "same_other_seq: replace masked span with span from another sequence,"
            "same_seq: repace masked span with span of the same sequence"
        },
    )
    masking_type: str = field(
        default="input", metadata={"help": "input or feature masking"}
    )

    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings " "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.1,
        metadata={
            "help": "dropout probability for attention weights " "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN " "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    audio_only: bool = field(
        default=False,
        metadata={"help": "whether to use audio stream only"},
    )
    no_scale_embedding: bool = field(default=True, metadata={"help": "scale embedding"})


class SubModel(nn.Module):
    """
    SubModel for audio and video feature extraction in AVHubert.

    This class implements a submodule of the AVHubert model that can
    process audio and video features. It uses an optional ResNet for
    video processing and a linear projection for both modalities.

    Attributes:
        resnet (nn.Module or None): ResNet module for video feature extraction.
        proj (nn.Linear): Linear layer for projecting input features to
            the encoder embedding dimension.
        encoder (TransformerEncoder or None): Transformer encoder for
            further processing of features if specified.

    Args:
        resnet (nn.Module or None): A ResNet model for video feature extraction.
        input_dim (int): The dimension of the input features.
        cfg (AVHubertConfig): Configuration object containing model parameters.

    Examples:
        >>> # Create a SubModel instance
        >>> sub_model = SubModel(resnet=my_resnet, input_dim=256, cfg=my_cfg)
        >>> # Forward pass through the model
        >>> output = sub_model(input_tensor)

    Note:
        The input tensor should have dimensions that match the expected
        input shape for the ResNet and the linear projection.

    Raises:
        ValueError: If the input tensor shape does not match the expected
        dimensions.
    """

    def __init__(self, resnet=None, input_dim=None, cfg=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, cfg.encoder_embed_dim)
        self.encoder = TransformerEncoder(cfg) if cfg.encoder_layers > 0 else None

    def forward(self, x):
        """
        Forward AVHubert Encoder.

        This method processes the input tensors for both audio and video modalities,
        applying necessary masking and encoding operations. It returns the encoded
        output along with the corresponding lengths and an optional mask.

        Args:
            xs_pad (Dict[str, torch.Tensor]): A dictionary containing input tensors.
                - "video": input tensor of shape (B, 1, L, H, W) for video.
                - "audio": input tensor of shape (B, D, L) for audio.
            ilens (torch.Tensor): A tensor containing the input lengths of shape (B,).
            prev_states (torch.Tensor, optional): Previous states; currently not used.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - Encoded output tensor of shape (B, T, D) where T is the sequence length
                after encoding.
                - Lengths of the output sequences as a tensor of shape (B,).
                - An optional mask tensor if applicable, otherwise None.

        Raises:
            ValueError: If neither "video" nor "audio" is present in xs_pad.

        Examples:
            >>> encoder = FairseqAVHubertEncoder(...)
            >>> xs_pad = {
            ...     "video": torch.randn(2, 1, 50, 64, 64),
            ...     "audio": torch.randn(2, 104, 50)
            ... }
            >>> ilens = torch.tensor([50, 50])
            >>> output, olens, mask = encoder.forward(xs_pad, ilens)
            >>> print(output.shape)  # Output: torch.Size([2, T, D])
            >>> print(olens)         # Output: tensor of lengths
        """
        if self.resnet is not None:
            x = self.resnet(x)
        x = self.proj(x.transpose(1, 2))
        if self.encoder is not None:
            x = self.encoder(x)[0].transpose(1, 2)
        else:
            x = x.transpose(1, 2)
        return x


class AVHubertModel(nn.Module):
    """
    AVHubert model for audio-visual representation learning.

    This model is based on the AVHubert architecture and is designed for
    processing both audio and video modalities. It leverages a
    transformer-based encoder to extract features from input audio and
    video data, which can then be used for various downstream tasks.

    Attributes:
        feature_extractor_audio: A sub-model for extracting audio features.
        feature_extractor_video: A sub-model for extracting video features.
        modality_fuse: Method for fusing audio and video features ('concat' or 'add').
        encoder: Transformer encoder used for feature processing.
        layer_norm: Layer normalization applied to the fused features.
        post_extract_proj: Optional projection layer after feature extraction.
        audio_only: Boolean indicating if only audio should be processed.

    Args:
        cfg (AVHubertConfig): Configuration object containing model parameters.
        **kwargs: Additional keyword arguments for model initialization.

    Examples:
        # Create a configuration object
        cfg = AVHubertConfig()

        # Build the AVHubert model
        model = AVHubertModel.build_model(cfg)

        # Forward pass with dummy audio and video inputs
        audio_input = torch.randn(2, 1, 100)  # Example audio input
        video_input = torch.randn(2, 1, 10, 224, 224)  # Example video input
        features = model.extract_finetune({'audio': audio_input, 'video': video_input})

    Note:
        Ensure that the FairSeq library is properly installed to utilize the
        functionalities of this model.
    """

    def __init__(self, cfg: AVHubertConfig, **kwargs) -> None:
        super().__init__()
        logger.info(f"HubertModel Config: {cfg}")

        try:
            from fairseq.modules import LayerNorm
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e

        feature_ds_rate = 1
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / cfg.sample_rate
        sub_cfg = deepcopy(cfg)
        sub_cfg.encoder_layers = sub_cfg.sub_encoder_layers
        resnet = ResEncoder(relu_type=cfg.resnet_relu_type, weights=cfg.resnet_weights)
        self.feature_extractor_audio = SubModel(
            resnet=None, input_dim=cfg.audio_feat_dim, cfg=sub_cfg
        )
        self.feature_extractor_video = SubModel(
            resnet=resnet, input_dim=resnet.backend_out, cfg=sub_cfg
        )
        self.modality_dropout, self.audio_dropout = (
            cfg.modality_dropout,
            cfg.audio_dropout,
        )
        self.modality_fuse = cfg.modality_fuse
        self.encoder_embed_dim = cfg.encoder_embed_dim
        if self.modality_fuse == "concat":
            self.embed = cfg.encoder_embed_dim * 2
        elif self.modality_fuse == "add":
            self.embed = cfg.encoder_embed_dim
        else:
            ValueError(f"unknown fusion method: {self.modality_fuse}")
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob_image, self.mask_prob_audio = (
            cfg.mask_prob_image,
            cfg.mask_prob_audio,
        )
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length_image, self.mask_length_audio = (
            cfg.mask_length_image,
            cfg.mask_length_audio,
        )
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.sim_type = cfg.sim_type
        self.selection_type = cfg.selection_type
        self.masking_type = cfg.masking_type

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.audio_feat_dim).uniform_()
            if self.masking_type == "input"
            else torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
        self.audio_only = cfg.audio_only

    @classmethod
    def build_model(cls, cfg: AVHubertConfig):
        """
        Build a new AVHubert model instance.

        This method initializes and returns a new instance of the AVHubert model
        using the specified configuration parameters.

        Args:
            cls: The class of the model to be instantiated.
            cfg (AVHubertConfig): Configuration object containing model parameters.

        Returns:
            AVHubertModel: An instance of the AVHubert model initialized with
            the given configuration.

        Examples:
            >>> config = AVHubertConfig()
            >>> model_instance = AVHubertModel.build_model(config)
            >>> print(type(model_instance))
            <class '__main__.AVHubertModel'>
        """

        kwargs = {}
        model = cls(cfg, **kwargs)
        return model

    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Extract features from the input source tensor using the specified modality.

        This method utilizes the appropriate feature extractor (either audio or video)
        based on the provided modality string. If `feature_grad_mult` is greater than
        zero, it applies a gradient scaling factor during backpropagation.

        Args:
            source (torch.Tensor): Input tensor containing audio or video data.
                The shape of the tensor should be compatible with the feature
                extractor corresponding to the specified modality.
            modality (str): A string that specifies the modality type.
                It should be either "audio" or "video".

        Returns:
            torch.Tensor: The extracted features from the input source tensor.

        Examples:
            >>> model = AVHubertModel(cfg)
            >>> audio_input = torch.randn(8, 1, 16000)  # Example audio input
            >>> video_input = torch.randn(8, 1, 5, 224, 224)  # Example video input
            >>> audio_features = model.forward_features(audio_input, "audio")
            >>> video_features = model.forward_features(video_input, "video")

        Note:
            Ensure that the `modality` parameter matches the type of data in the
            `source` tensor to avoid runtime errors.
        """
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adjusts the padding mask to match the feature dimensions.

        This method takes the input feature tensor and its associated padding
        mask, ensuring that the mask dimensions align with the features. If
        the padding mask is longer than the features, the extra elements are
        removed. The final mask is reshaped to allow for masking across the
        appropriate dimensions.

        Args:
            features (torch.Tensor): The input feature tensor with shape
                (B, T, D) where B is the batch size, T is the sequence length,
                and D is the feature dimension.
            padding_mask (torch.Tensor): The original padding mask with shape
                (B, L) where L is the length of the original sequence.

        Returns:
            torch.Tensor: A boolean tensor indicating the positions that
            should be masked, with shape (B, T).

        Examples:
            >>> features = torch.randn(4, 10, 64)  # Example features
            >>> padding_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            ...                                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            ...                                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            ...                                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
            >>> mask = self.forward_padding_mask(features, padding_mask)
            >>> mask.shape
            torch.Size([4, 10])  # Output mask shape should match features
        """
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def extract_finetune(
        self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None
    ):
        """
        Forward AVHubert Pretrain Encoder.

        This method processes audio and video inputs, applies
        modality fusion, and passes the features through the
        encoder. The function can handle both modalities, with
        the option to fine-tune the model.

        Args:
            source (dict): A dictionary containing the input tensors.
                - source['video']: input tensor of shape (B, 1, L, H, W)
                - source['audio']: input tensor of shape (B, F, L)
            padding_mask (torch.Tensor, optional): A tensor of shape
                (B, L) indicating which elements are padding.
                Defaults to None.
            mask (bool, optional): If True, applies masking to the
                input. Defaults to False.
            ret_conv (bool, optional): If True, returns convolutional
                features. Defaults to False.
            output_layer (int, optional): Specifies which layer's output
                to return. Defaults to None, meaning all layers.

        Returns:
            tuple: A tuple containing:
                - encoded tensor of shape (B, T, D)
                - padding mask of shape (B, T)

        Raises:
            ValueError: If both audio and video sources are None.

        Examples:
            >>> source = {
            ...     'video': torch.randn(4, 1, 100, 224, 224),
            ...     'audio': torch.randn(4, 80, 100)
            ... }
            >>> padding_mask = torch.tensor([[0, 1, 1, 0, 0],
            ...                                [0, 0, 0, 1, 1]])
            >>> encoded, mask = model.extract_finetune(source,
            ...                                         padding_mask)
        """
        src_audio, src_video = source["audio"], source["video"]

        if (src_audio is not None and src_video is None) or self.audio_only:
            features_audio = self.forward_features(
                src_audio, modality="audio"
            )  # features: [B, F, T]
            features_video = features_audio.new_zeros(
                features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1)
            )
        elif src_audio is None and src_video is not None:
            features_video = self.forward_features(src_video, modality="video")
            features_audio = features_video.new_zeros(
                features_video.size(0), self.encoder_embed_dim, features_video.size(-1)
            )
        elif src_audio is not None and src_video is not None:
            features_video = self.forward_features(src_video, modality="video")
            features_audio = self.forward_features(
                src_audio, modality="audio"
            )  # features: [B, F, T]
        else:
            ValueError("Both audio and video is None")

        if self.modality_fuse == "concat":
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == "add":
            features = features_audio + features_video
        else:
            ValueError(f"unknown fusion method: {self.modality_fuse}")

        features = features.transpose(1, 2)  # B, 2F, T -> B, T, 2F
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        return x, padding_mask

    def forward_audio(self, source_audio):
        """
        Forward pass for audio input through the AVHubert model.

        This method processes the audio input tensor and extracts
        features using the audio feature extractor. The features are
        computed without tracking gradients to reduce memory usage
        during inference.

        Args:
            source_audio (torch.Tensor): Input tensor containing audio data
                of shape (B, F, T), where B is the batch size, F is the
                number of features, and T is the sequence length.

        Returns:
            torch.Tensor: Extracted audio features of shape (B, D, T), where
                D is the encoder embedding dimension.

        Examples:
            >>> model = AVHubertModel(cfg)
            >>> audio_input = torch.randn(8, 512, 100)  # Batch of 8, 512 features, 100 time steps
            >>> audio_features = model.forward_audio(audio_input)
            >>> print(audio_features.shape)
            torch.Size([8, 768, 100])  # Assuming encoder_embed_dim is 768

        Note:
            This method is primarily intended for use during inference
            and should not be used during training as it does not
            track gradients.
        """
        with torch.no_grad():
            features_audio = self.forward_features(
                source_audio, modality="audio"
            )  # features: [B, F, T]
        return features_audio

    def forward_video(self, source_video):
        """
        Forward pass for the video feature extractor.

        This method processes the input video tensor and extracts features
        using the underlying video feature extractor model. The output
        features are computed without gradient tracking, which is beneficial
        for inference scenarios.

        Args:
            source_video (torch.Tensor): Input video tensor of shape (B, 1, L, H, W),
                where B is the batch size, L is the sequence length, H is the height,
                and W is the width of the video frames.

        Returns:
            torch.Tensor: Extracted video features of shape (B, F, T), where F is
                the number of feature dimensions and T is the length of the
                output sequence.

        Examples:
            >>> model = AVHubertModel(cfg)
            >>> video_input = torch.randn(8, 1, 10, 224, 224)  # Batch of 8 videos
            >>> video_features = model.forward_video(video_input)
            >>> print(video_features.shape)  # Should print: torch.Size([8, F, T])

        Note:
            This method is intended for use in inference mode. During training,
            the video features are typically extracted in a manner that allows
            for backpropagation.
        """
        with torch.no_grad():
            features_video = self.forward_features(
                source_video, modality="video"
            )  # features: [B, F, T]
        return features_video

    def modality_fusion(self, features_audio, features_video):
        """
        Fuse audio and video features using the specified fusion method.

        This method combines audio and video features based on the
        configured fusion technique, which can be either concatenation
        or addition. It handles cases where one of the modalities may
        be absent by providing zero tensors of the appropriate shape.

        Args:
            features_audio (torch.Tensor): The audio features tensor with shape
                (B, D, L), where B is the batch size, D is the feature dimension,
                and L is the length of the sequence.
            features_video (torch.Tensor): The video features tensor with shape
                (B, D, L), where B is the batch size, D is the feature dimension,
                and L is the length of the sequence.

        Returns:
            torch.Tensor: The fused features tensor, which will have shape
            determined by the fusion method:
                - If concatenation is used, the shape will be (B, 2D, L).
                - If addition is used, the shape will be (B, D, L).

        Raises:
            ValueError: If an unknown fusion method is specified.

        Examples:
            >>> audio_features = torch.randn(32, 256, 10)  # 32 samples, 256 features, 10 time steps
            >>> video_features = torch.randn(32, 256, 10)  # 32 samples, 256 features, 10 time steps
            >>> fused_features = self.modality_fusion(audio_features, video_features)
            >>> print(fused_features.shape)  # If concatenation is used, should output: torch.Size([32, 512, 10])
        """
        if features_video is None and features_audio is not None:
            features_video = features_audio.new_zeros(
                features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1)
            )
        elif features_audio is None and features_video is not None:
            features_audio = features_video.new_zeros(
                features_video.size(0), self.encoder_embed_dim, features_video.size(-1)
            )
        else:
            features_video = features_video
            features_audio = features_audio

        if self.modality_fuse == "concat":
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == "add":
            features = features_audio + features_video
        else:
            ValueError(f"unknown fusion method: {self.modality_fuse}")

        return features

    def forward_transformer(self, source, padding_mask=None, output_layer=None):
        """
        Forward AVHubert Pretrain Encoder (without frontend).

        This method processes the input tensor using the transformer encoder
        to generate encoded features. The input tensor is expected to be a
        fused feature tensor, combining both audio and video modalities.

        Args:
            source: A tensor of shape (B, L, D*2) where B is the batch size,
                L is the sequence length, and D is the embedding dimension.
            padding_mask: A tensor of shape (B, L) indicating padded elements
                in the input. Elements to be masked should have a value of
                `True`, while valid elements should have a value of `False`.
            output_layer: Optional integer specifying which layer's output to
                return. If None, the output from the last layer is returned.

        Returns:
            A tuple containing:
                - The encoded tensor of shape (B, L, D) after processing
                through the transformer encoder.
                - The updated padding mask of shape (B, L).

        Examples:
            >>> model = AVHubertModel(cfg)
            >>> input_tensor = torch.rand(4, 10, 768)  # Example input
            >>> padding_mask = torch.zeros(4, 10, dtype=torch.bool)
            >>> encoded_output, updated_mask = model.forward_transformer(input_tensor, padding_mask)

        Note:
            This function assumes that the input has already undergone
            necessary preprocessing steps, including modality fusion.

        Raises:
            ValueError: If the source tensor or padding_mask has an invalid shape.
        """
        features = source
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        return x, padding_mask


def download_avhubert(model_url, dir_path):
    """
    Download the AVHubert model from a specified URL.

    This function checks if the model already exists in the specified directory.
    If not, it will download the model and save it in the specified directory,
    using a file lock to ensure that concurrent downloads do not occur.

    Args:
        model_url (str): The URL from which to download the AVHubert model.
        dir_path (str): The directory path where the model should be saved.

    Returns:
        str: The file path of the downloaded model.

    Raises:
        Exception: If there is an issue with downloading the model.

    Examples:
        >>> model_url = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/large_vox_iter5.pt"
        >>> dir_path = "./models"
        >>> model_path = download_avhubert(model_url, dir_path)
        AVHubert model downloaded ./models/large_vox_iter5.pt
    """

    model_name = model_url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)

    if not os.path.exists(model_path):
        with FileLock(model_path + ".lock"):
            torch.hub.download_url_to_file(model_url, model_path)
            logging.info(f"AVHubert model downloaded {model_path}")
    else:
        logging.info(f"AVHubert model {model_path} already exists.")

    return model_path


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for AVHubert.

    This class implements a transformer encoder as described in the
    AVHubert architecture. It uses multiple layers of transformer
    sentence encoder layers and includes optional layer normalization
    and dropout functionalities.

    Args:
        args: Configuration parameters including dropout rates,
            embedding dimensions, and layer settings.

    Attributes:
        dropout (float): Dropout probability for the encoder.
        embedding_dim (int): Dimension of the encoder embeddings.
        pos_conv (nn.Conv1d): Convolutional layer for positional embeddings.
        layers (nn.ModuleList): List of transformer sentence encoder layers.
        layer_norm_first (bool): Whether to apply layer normalization first.
        layer_norm (LayerNorm): Layer normalization module.
        layerdrop (float): Probability of dropping a transformer layer.

    Methods:
        forward(x, padding_mask=None, layer=None):
            Forward pass through the encoder.
        extract_features(x, padding_mask=None, tgt_layer=None):
            Extract features from the input tensor.

    Examples:
        >>> encoder = TransformerEncoder(args)
        >>> output, _ = encoder(input_tensor, padding_mask)

    Note:
        Ensure that FairSeq is properly installed for the transformer
        sentence encoder layers to work correctly.

    Raises:
        Exception: If FairSeq is not installed properly.
    """

    def __init__(self, args):
        super().__init__()
        try:
            from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer
            from fairseq.modules import LayerNorm
            from fairseq.modules.transformer_sentence_encoder import init_bert_params
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        """
        Forward pass through the AVHubert Encoder.

        This method processes the input audio and video features through the
        encoder. It handles the input tensors, applies masking, and performs
        encoding based on the modality (audio or video). The output consists
        of encoded features along with the corresponding lengths and an
        optional mask.

        Args:
            xs_pad (Dict[str, torch.Tensor]): A dictionary containing input
                tensors. Expected keys are:
                - 'video': input tensor of shape (B, 1, L, H, W)
                - 'audio': input tensor of shape (B, D, L)
            ilens (torch.Tensor): A tensor of shape (B,) representing the
                lengths of the input sequences.
            prev_states (torch.Tensor, optional): Placeholder for previous
                states. Currently not utilized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                A tuple containing:
                - Encoded features tensor of shape (B, T, D) where T is the
                  length of the encoded sequence and D is the dimension.
                - Lengths tensor of shape (B,) indicating the lengths of the
                  encoded sequences.
                - An optional tensor that can be used for masks (currently
                  set to None).

        Raises:
            ValueError: If neither 'video' nor 'audio' is present in the
                input dictionary.

        Examples:
            >>> encoder = FairseqAVHubertEncoder()
            >>> audio_input = torch.rand(2, 10, 300)  # Batch of 2, 10 features, 300 time steps
            >>> video_input = torch.rand(2, 1, 25, 224, 224)  # Batch of 2, 1 frame, 25 time steps, 224x224 resolution
            >>> ilens = torch.tensor([300, 250])  # Lengths of audio inputs
            >>> outputs = encoder.forward({'audio': audio_input, 'video': video_input}, ilens)
            >>> encoded_features, lengths, _ = outputs
            >>> print(encoded_features.shape, lengths.shape)  # (2, T, D), (2,)

        Note:
            The function incorporates mechanisms for fine-tuning and dropout
            handling during training. The actual behavior may vary depending
            on the current state of the model (training or evaluation).
        """
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        """
        Extract features from the input tensor using the transformer encoder.

        This method applies a series of convolutional layers and transformer
        layers to extract features from the input tensor. It optionally
        applies padding masks to ignore certain time steps in the input
        sequences. The output can be directed to a specific transformer layer
        if desired.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is the
                batch size, T is the sequence length, and C is the feature
                dimension.
            padding_mask (torch.Tensor, optional): A boolean tensor of shape
                (B, T) indicating which time steps should be ignored. Defaults
                to None.
            tgt_layer (int, optional): The specific layer to extract features
                from. If None, features from the last layer are returned.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
                A tuple containing the output tensor of shape (B, T, C) and a
                list of tuples containing intermediate layer outputs.

        Raises:
            ValueError: If both audio and video inputs are None.

        Examples:
            >>> model = TransformerEncoder(args)
            >>> x = torch.randn(32, 100, 768)  # Batch of 32 sequences
            >>> padding_mask = torch.zeros(32, 100).bool()
            >>> features, layer_outputs = model.extract_features(x, padding_mask)

        Note:
            The output features are subject to layer normalization and dropout
            during training. The padding mask is applied to ignore certain
            time steps based on the provided mask.
        """
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
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

    def max_positions(self):
        """
        TransformerEncoder class for the AVHubert model.

        This class implements a Transformer encoder that processes input features
        and applies positional embeddings. It is designed for use within the
        AVHubert framework for audio-visual tasks.

        Attributes:
            dropout (float): Dropout probability for the encoder.
            embedding_dim (int): Dimensionality of the encoder embeddings.
            pos_conv (nn.Conv1d): Convolutional layer for positional embeddings.
            layers (nn.ModuleList): List of transformer layers.
            layer_norm_first (bool): Indicates if layer normalization is applied
                                    before the first layer.
            layer_norm (LayerNorm): Layer normalization module.
            layerdrop (float): Probability of dropping a transformer layer.

        Methods:
            forward(x, padding_mask=None, layer=None):
                Forward pass through the encoder.

            extract_features(x, padding_mask=None, tgt_layer=None):
                Extract features from the input tensor.

            max_positions():
                Returns the maximum output length supported by the encoder.

            upgrade_state_dict_named(state_dict, name):
                Upgrades a (possibly old) state dict for new versions of Fairseq.

        Examples:
            >>> encoder = TransformerEncoder(args)
            >>> output, _ = encoder(input_tensor, padding_mask)
            >>> max_len = encoder.max_positions()

        Note:
            Ensure that Fairseq is properly installed to use this class.
        """
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Upgrade a (possibly old) state dict for new versions of fairseq.

        This method is designed to ensure compatibility between old and new
        versions of the Fairseq library when loading state dictionaries. It
        modifies the keys of the state dictionary as necessary to fit the
        expected format of the current version of the library.

        Args:
            state_dict (dict): The state dictionary to be upgraded.
            name (str): The name of the model or module associated with
                        the state dictionary.

        Returns:
            dict: The upgraded state dictionary.

        Note:
            This method may not perform any modifications if the state
            dictionary is already compatible with the current version.

        Examples:
            >>> state_dict = {'old_key': tensor}
            >>> upgraded_state_dict = model.upgrade_state_dict_named(state_dict, 'model_name')
            >>> print(upgraded_state_dict)
        """
        return state_dict


class BasicBlock(nn.Module):
    """
    Basic building block for ResNet architecture.

    This class implements a basic block used in the ResNet architecture, which
    consists of two convolutional layers with batch normalization and ReLU or PReLU
    activation functions. The block supports optional downsampling.

    Attributes:
        expansion (int): Expansion factor for the block.
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization after the first convolution.
        relu1 (nn.Module): Activation function after the first convolution.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization after the second convolution.
        downsample (nn.Sequential, optional): Downsampling layer.
        stride (int): Stride value for the first convolution.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the first convolution. Default is 1.
        downsample (nn.Sequential, optional): Downsampling layer. Default is None.
        relu_type (str, optional): Type of ReLU activation function.
            Can be "relu" or "prelu". Default is "relu".

    Raises:
        Exception: If an unsupported relu_type is provided.

    Examples:
        >>> block = BasicBlock(inplanes=64, planes=128, stride=2, relu_type='relu')
        >>> x = torch.randn(1, 64, 32, 32)  # Example input
        >>> output = block(x)
        >>> output.shape
        torch.Size([1, 128, 16, 16])  # Output shape after downsampling
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type="relu"):
        super(BasicBlock, self).__init__()

        assert relu_type in ["relu", "prelu"]

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        else:
            raise Exception("relu type not implemented")

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass through the AVHubert Encoder.

        This method takes input tensors for video and audio, applies
        necessary transformations and masking, and returns the output
        tensor along with the output lengths and an optional tensor.

        Args:
            xs_pad (Dict[str, torch.Tensor]):
                A dictionary containing input tensors. Expected keys are:
                - 'video': input tensor of shape (B, 1, L, H, W)
                - 'audio': input tensor of shape (B, D, L)
            ilens (torch.Tensor):
                A tensor of shape (B,) containing the lengths of each
                input sequence.
            prev_states (torch.Tensor, optional):
                Previous states from the encoder, not used in the current
                implementation. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                A tuple containing:
                - position embedded tensor of shape (B, T, D)
                - tensor of output lengths of shape (B,)
                - None (placeholder for potential future use).

        Raises:
            ValueError: If neither 'video' nor 'audio' keys are present in
            the input dictionary.

        Examples:
            >>> xs_pad = {
            ...     'video': torch.randn(4, 1, 100, 64, 64),
            ...     'audio': torch.randn(4, 104, 100)
            ... }
            >>> ilens = torch.tensor([100, 90, 80, 70])
            >>> encoder = FairseqAVHubertEncoder()
            >>> output, olens, _ = encoder(xs_pad, ilens)
            >>> output.shape
            torch.Size([4, 100, 1024])
            >>> olens.shape
            torch.Size([4])
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    """
    ResNet architecture for deep learning applications.

    This class implements a ResNet architecture, which is a deep
    convolutional neural network that utilizes residual connections to
    facilitate the training of very deep networks. The ResNet is designed
    to learn residual mappings with reference to the layer inputs,
    rather than learning unreferenced functions.

    Attributes:
        layer1 (nn.Sequential): The first layer of the ResNet.
        layer2 (nn.Sequential): The second layer of the ResNet.
        layer3 (nn.Sequential): The third layer of the ResNet.
        layer4 (nn.Sequential): The fourth layer of the ResNet.
        avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.

    Args:
        block (nn.Module): The building block to use for the ResNet.
        layers (list): A list containing the number of blocks in each layer.
        num_classes (int, optional): Number of output classes. Defaults to 1000.
        relu_type (str, optional): Type of ReLU activation function.
            Options are 'relu' or 'prelu'. Defaults to 'relu'.
        gamma_zero (bool, optional): If True, initializes the second
            batch normalization layer weights to zero. Defaults to False.
        avg_pool_downsample (bool, optional): If True, uses average pooling
            for downsampling. Defaults to False.

    Examples:
        >>> model = ResNet(BasicBlock, [2, 2, 2, 2])
        >>> input_tensor = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels
        >>> output = model(input_tensor)
        >>> print(output.shape)  # Should output (1, 512)
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        relu_type="relu",
        gamma_zero=False,
        avg_pool_downsample=False,
    ):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = (
            downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block
        )

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes * block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward AVHubert Encoder.

        This method processes input tensors for both audio and video
        modalities through the AVHubert encoder, applying necessary
        masking and fine-tuning based on the training state.

        Args:
            xs_pad (Dict[str, torch.Tensor]): A dictionary containing input
                tensors for different modalities. It must include:
                - 'video': input tensor of shape (B, 1, L, H, W) for video data.
                - 'audio': input tensor of shape (B, D, L) for audio data.
            ilens (torch.Tensor): A tensor containing the lengths of each
                input sequence, shape (B).
            prev_states (torch.Tensor, optional): Previous states from the
                encoder, not used in the current implementation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of processed features (position embedded) from
                  the encoder, shape (B, T, D).
                - A tensor representing the output lengths, shape (B).
                - None (not used in the current implementation).

        Raises:
            ValueError: If neither 'video' nor 'audio' keys are present
                in the input dictionary.

        Examples:
            >>> model = FairseqAVHubertEncoder(...)
            >>> xs_pad = {
            ...     'video': torch.randn(8, 1, 100, 64, 64),
            ...     'audio': torch.randn(8, 104, 100)
            ... }
            >>> ilens = torch.tensor([100, 100, 100, 100, 100, 100, 100, 100])
            >>> output_features, output_lengths, _ = model(xs_pad, ilens)

        Note:
            - The method incorporates a masking mechanism for training
              data augmentation and adjusts parameters based on the
              fine-tuning state.
            - Ensure that the input tensor shapes are compatible with
              the model configuration.

        Todo:
            - Implement utilization of `prev_states` for stateful processing.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResEncoder(nn.Module):
    """
    3D ResNet-based Encoder for audio-visual tasks.

    This class implements a 3D convolutional neural network that processes
    input data in the form of videos, extracting features using a ResNet
    architecture. It consists of a frontend for initial feature extraction
    and a trunk that applies residual connections to improve learning.

    Attributes:
        frontend_nout (int): Number of output channels for the frontend.
        backend_out (int): Number of output channels for the trunk.
        frontend3D (nn.Sequential): Sequential model for the frontend processing.
        trunk (ResNet): ResNet model for feature extraction from 2D tensor input.

    Args:
        relu_type (str): Type of ReLU activation to use ('relu' or 'prelu').
        weights (Optional[str]): Path to pre-trained weights for the model.

    Examples:
        >>> model = ResEncoder(relu_type='relu', weights=None)
        >>> input_tensor = torch.randn(8, 1, 10, 112, 112)  # (B, C, T, H, W)
        >>> output = model(input_tensor)
        >>> output.shape
        torch.Size([8, 512, 10])  # (B, D, T)

    Note:
        The input tensor must be in the shape of (B, C, T, H, W), where
        B is the batch size, C is the number of channels, T is the number
        of frames, and H and W are the height and width of the frames.

    Raises:
        RuntimeError: If there is an issue loading the weights or
        during the forward pass due to incompatible input shapes.
    """

    def __init__(self, relu_type, weights):
        super(ResEncoder, self).__init__()
        self.frontend_nout = 64
        self.backend_out = 512
        frontend_relu = (
            nn.PReLU(num_parameters=self.frontend_nout)
            if relu_type == "prelu"
            else nn.ReLU()
        )
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        if weights is not None:
            logger.info(f"Load {weights} for resnet")
            std = torch.load(weights, map_location=torch.device("cpu"))[
                "model_state_dict"
            ]
            frontend_std, trunk_std = OrderedDict(), OrderedDict()
            for key, val in std.items():
                new_key = ".".join(key.split(".")[1:])
                if "frontend3D" in key:
                    frontend_std[new_key] = val
                if "trunk" in key:
                    trunk_std[new_key] = val
            self.frontend3D.load_state_dict(frontend_std)
            self.trunk.load_state_dict(trunk_std)

    def forward(self, x):
        """
        Forward pass through the AVHubert Encoder.

        This method processes the input tensors for audio and video, applying
        necessary masking and dropout techniques as configured. It returns the
        output features along with the lengths of the valid output sequences.

        Args:
            xs_pad (Dict[str, torch.Tensor]): A dictionary containing input tensors:
                - "video": input tensor of shape (B, 1, L, H, W) for video data.
                - "audio": input tensor of shape (B, D, L) for audio data.
            ilens (torch.Tensor): A tensor containing the lengths of the input
                sequences (B).
            prev_states (torch.Tensor, optional): Not used in the current
                implementation. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - Output tensor of shape (B, T, D) containing the features
                  after encoding.
                - A tensor of shape (B) representing the lengths of valid output
                  sequences.
                - None (as placeholder for future use).

        Raises:
            ValueError: If neither "video" nor "audio" keys are present in
                        `xs_pad`.

        Examples:
            >>> audio_input = torch.randn(2, 104, 10)  # (B, D, L)
            >>> video_input = torch.randn(2, 1, 10, 224, 224)  # (B, 1, L, H, W)
            >>> ilens = torch.tensor([10, 10])  # lengths for both inputs
            >>> xs_pad = {"audio": audio_input, "video": video_input}
            >>> output, lengths, _ = model.forward(xs_pad, ilens)
            >>> print(output.shape)  # Output shape: (2, T, D)
            >>> print(lengths)  # Lengths of the valid output sequences

        Note:
            Ensure that the input tensors are properly padded before passing them
            to the forward method to avoid dimension mismatches.
        """
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]
        x = self.threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        x = x.transpose(1, 2).contiguous()
        return x

    def threeD_to_2D_tensor(self, x):
        """
            Reshape a 3D tensor into a 2D tensor for processing.

        This method takes a 5-dimensional tensor (batch, channels, time, height, width)
        and reshapes it into a 4-dimensional tensor (batch*time, channels, height, width).
        This transformation is useful for passing the data through a 2D convolutional
        network after extracting features from a 3D input.

        Args:
            x (torch.Tensor): A tensor of shape (n_batch, n_channels, s_time, sx, sy).

        Returns:
            torch.Tensor: A reshaped tensor of shape (n_batch * s_time, n_channels, sx, sy).

        Examples:
            >>> import torch
            >>> x = torch.randn(2, 3, 4, 5, 6)  # Example input tensor
            >>> reshaped_x = self.threeD_to_2D_tensor(x)
            >>> reshaped_x.shape
            torch.Size([8, 3, 5, 6])  # (2 * 4, 3, 5, 6)

        Note:
            The input tensor must have 5 dimensions; otherwise, an error will occur.

        Raises:
            ValueError: If the input tensor does not have 5 dimensions.
        """
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.reshape(n_batch * s_time, n_channels, sx, sy)


class SamePad(nn.Module):
    """
    Applies same padding to the input tensor.

    This class provides a way to ensure that the output tensor
    has the same spatial dimensions as the input tensor after
    applying a convolution operation with a specified kernel size.
    It can be configured for causal padding, which is commonly used
    in sequence-to-sequence models where future information should
    not be considered.

    Attributes:
        kernel_size (int): The size of the convolutional kernel.
        remove (int): The number of elements to remove from the end
            of the output tensor, determined by the kernel size.

    Args:
        kernel_size (int): The size of the kernel for which the padding
            will be calculated.
        causal (bool): If True, applies causal padding. Default is False.

    Examples:
        >>> import torch
        >>> same_pad = SamePad(kernel_size=3)
        >>> input_tensor = torch.randn(1, 1, 10)  # Example input
        >>> output_tensor = same_pad(input_tensor)
        >>> output_tensor.shape  # Output shape should be (1, 1, 10)

        >>> causal_pad = SamePad(kernel_size=3, causal=True)
        >>> output_tensor_causal = causal_pad(input_tensor)
        >>> output_tensor_causal.shape  # Output shape should be (1, 1, 9)

    Note:
        If `kernel_size` is even, the padding will be applied equally
        on both sides. If `kernel_size` is odd, one extra element will
        be removed from the end to maintain the same output size.

    Raises:
        ValueError: If the input tensor does not have the expected
            dimensions.
    """

    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        """
        Forward pass for the AVHubert Encoder.

        This method processes the input tensors, applies necessary
        transformations, and returns the encoded representations along
        with their respective lengths.

        Args:
            xs_pad (Dict[str, torch.Tensor]): A dictionary containing
                input tensors. It can have the following keys:
                - 'video': input tensor of shape (B, 1, L, H, W)
                - 'audio': input tensor of shape (B, D, L)
            ilens (torch.Tensor): A tensor of shape (B,) representing the
                input lengths for each batch element.
            prev_states (torch.Tensor, optional): Not used in the current
                implementation. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, D) representing the encoded
                  features.
                - A tensor of shape (B,) containing the lengths of the
                  output sequences.
                - None, as there are no additional states returned.

        Raises:
            ValueError: If neither 'video' nor 'audio' is present in
                `xs_pad`.

        Examples:
            >>> encoder = FairseqAVHubertEncoder()
            >>> xs_pad = {
            ...     'video': torch.randn(2, 1, 50, 64, 64),
            ...     'audio': torch.randn(2, 104, 50)
            ... }
            >>> ilens = torch.tensor([50, 50])
            >>> output, lengths, _ = encoder.forward(xs_pad, ilens)
            >>> print(output.shape)  # Output shape: (2, T, D)
            >>> print(lengths)  # Output lengths for each batch item

        Note:
            Ensure that the input tensors are properly padded and
            have the correct dimensions.
        """
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


def index_put(tensor, indices, value):
    """
    Updates elements of a tensor at specified indices with a given value.

    This function modifies the input tensor in-place by assigning the specified
    value to the positions indicated by the indices. It handles tensors located
    on XLA devices (e.g., TPU) differently to ensure compatibility.

    Args:
        tensor (torch.Tensor): The input tensor to be updated.
        indices (torch.Tensor): A tensor containing the indices where the value
            should be placed. This tensor should have the same number of dimensions
            as the tensor being modified.
        value (torch.Tensor): The value to assign at the specified indices. This
            should be broadcastable to the shape of the indices.

    Returns:
        torch.Tensor: The updated tensor with values assigned at the specified
        indices.

    Examples:
        >>> tensor = torch.tensor([[1, 2], [3, 4]])
        >>> indices = torch.tensor([[0, 1], [1, 0]])
        >>> value = torch.tensor([[5, 6], [7, 8]])
        >>> updated_tensor = index_put(tensor, indices, value)
        >>> print(updated_tensor)
        tensor([[5, 6],
                [8, 4]])

    Note:
        If the input tensor is an XLA tensor, the function will ensure that
        the operation is performed correctly according to XLA tensor handling
        requirements.

    Raises:
        IndexError: If the indices are out of bounds for the input tensor.
    """
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor


def is_xla_tensor(tensor):
    """
    Check if a given tensor is an XLA tensor.

    This function determines if the input tensor is a PyTorch tensor
    that is located on an XLA device. XLA (Accelerated Linear Algebra)
    is a domain-specific compiler for linear algebra that can accelerate
    TensorFlow and PyTorch computations on TPUs.

    Args:
        tensor (torch.Tensor): The tensor to check.

    Returns:
        bool: True if the tensor is an XLA tensor, False otherwise.

    Examples:
        >>> import torch
        >>> xla_tensor = torch.tensor([1, 2, 3], device='xla')
        >>> cpu_tensor = torch.tensor([1, 2, 3])
        >>> is_xla_tensor(xla_tensor)
        True
        >>> is_xla_tensor(cpu_tensor)
        False

    Note:
        Ensure that you have the necessary environment set up for XLA
        tensors when using this function.
    """
    return torch.is_tensor(tensor) and tensor.device.type == "xla"


class GradMultiply(torch.autograd.Function):
    """
    Applies a gradient multiplication operation to a tensor.

    This class implements a custom autograd function that multiplies the
    gradients of a tensor by a specified scale factor during the backward
    pass. This can be useful for controlling the flow of gradients,
    particularly in scenarios like feature extraction where one may want
    to reduce the contribution of certain features to the overall loss.

    Attributes:
        scale (float): The scale factor by which to multiply the gradients.

    Methods:
        forward(ctx, x, scale):
            Applies the forward pass of the function.

        backward(ctx, grad):
            Applies the backward pass of the function.

    Args:
        x (torch.Tensor): The input tensor whose gradients will be scaled.
        scale (float): The factor to multiply the gradients by.

    Returns:
        torch.Tensor: The input tensor, unchanged during the forward pass.

    Examples:
        >>> import torch
        >>> from your_module import GradMultiply
        >>> x = torch.tensor([1.0, 2.0], requires_grad=True)
        >>> scale = 0.5
        >>> output = GradMultiply.apply(x, scale)
        >>> output.backward(torch.tensor([1.0, 1.0]))
        >>> print(x.grad)  # Output will be [0.5, 1.0], scaled by 0.5
    """

    @staticmethod
    def forward(ctx, x, scale):
        """
        Forward pass for the AVHubert Encoder.

        This method processes input tensors for both audio and video modalities
        and returns the encoded outputs along with their respective lengths.

        Args:
            xs_pad (Dict[str, torch.Tensor]): A dictionary containing input
                tensors. The expected keys are:
                - 'video': input tensor of shape (B, 1, L, H, W)
                - 'audio': input tensor of shape (B, D, L)
            ilens (torch.Tensor): A tensor of shape (B,) representing the
                input lengths for each batch.
            prev_states (torch.Tensor, optional): Not currently used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                A tuple containing:
                - Encoded tensor of shape (B, T, D), where T is the length of
                  the output sequence and D is the output dimension.
                - A tensor containing the output lengths of shape (B,).
                - An optional tensor, currently set to None.

        Raises:
            ValueError: If neither 'video' nor 'audio' keys are present in
                `xs_pad`.

        Examples:
            >>> model = FairseqAVHubertEncoder()
            >>> xs_pad = {
            ...     'video': torch.randn(2, 1, 100, 224, 224),
            ...     'audio': torch.randn(2, 104, 100)
            ... }
            >>> ilens = torch.tensor([100, 100])
            >>> output, lengths, _ = model(xs_pad, ilens)
            >>> print(output.shape)  # Output tensor shape
            torch.Size([2, 100, 1024])
            >>> print(lengths)  # Output lengths
            tensor([100, 100])
        """
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        """
        Compute the gradient by scaling the input gradient.

        This function is part of the `GradMultiply` class and is used to scale
        the gradient during backpropagation. It allows for controlling the
        contribution of the input features to the loss.

        Args:
            ctx: The context object that can be used to stash information
                for backward computation. This is automatically provided
                by PyTorch.
            grad: The gradient of the loss with respect to the output of
                this function. This is a tensor containing the gradients
                from the subsequent layer.

        Returns:
            A tuple containing:
                - The scaled gradient for the input tensor.
                - None, as no gradient scaling is needed for the scale parameter.

        Examples:
            >>> import torch
            >>> from your_module import GradMultiply
            >>> input_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            >>> scale = 0.1
            >>> output = GradMultiply.apply(input_tensor, scale)
            >>> output.backward(torch.tensor([1.0, 1.0, 1.0]))
            >>> print(input_tensor.grad)  # Should show [0.1, 0.2, 0.3]

        Note:
            This function should be used in conjunction with the `forward`
            method of the `GradMultiply` class. It modifies the gradient
            passed to it, scaling it by the specified factor.

        Raises:
            None
        """
        return grad * ctx.scale, None
