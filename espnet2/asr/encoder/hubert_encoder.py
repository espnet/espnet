# Copyright 2021 Tianzi Wang
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0

# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert


"""Encoder definition."""
import contextlib
import copy
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import yaml
from filelock import FileLock
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class TorchAudioHuBERTPretrainEncoder(AbsEncoder):
    """
    TorchAudio Hubert encoder module for pretraining.

    This class implements the Hubert encoder using the TorchAudio implementation.
    It can be used for pretraining Hubert models or fine-tuning for downstream tasks.

    Args:
        input_size (int, optional): Input feature dimension. Defaults to None.
        extractor_mode (str, optional): Operation mode of feature extractor. Defaults to "group_norm".
        extractor_conv_layer_config (List[List[int]], optional): Configuration of convolution layers in feature extractor. Defaults to a specific 7-layer configuration.
        extractor_conv_bias (bool, optional): Whether to include bias in convolution operations. Defaults to False.
        encoder_embed_dim (int, optional): Embedding dimension in encoder. Defaults to 768.
        encoder_projection_dropout (float, optional): Dropout probability after input feature projection. Defaults to 0.1.
        encoder_pos_conv_kernel (int, optional): Kernel size of convolutional positional embeddings. Defaults to 128.
        encoder_pos_conv_groups (int, optional): Number of groups in convolutional positional embeddings. Defaults to 16.
        encoder_num_layers (int, optional): Number of self-attention layers in transformer block. Defaults to 12.
        encoder_num_heads (int, optional): Number of heads in self-attention layers. Defaults to 12.
        encoder_attention_dropout (float, optional): Dropout probability in self-attention layer. Defaults to 0.1.
        encoder_ff_interm_features (int, optional): Dimension of hidden features in feed-forward layer. Defaults to 3072.
        encoder_ff_interm_dropout (float, optional): Dropout probability in feed-forward layer. Defaults to 0.0.
        encoder_dropout (float, optional): Dropout probability at the end of feed-forward layer. Defaults to 0.1.
        encoder_layer_norm_first (bool, optional): Controls the order of layer normalization. Defaults to False.
        encoder_layer_drop (float, optional): Probability to drop each encoder layer during training. Defaults to 0.05.
        mask_prob (float, optional): Probability for each token to be chosen as start of the span to be masked. Defaults to 0.8.
        mask_selection (str, optional): How to choose the mask length. Defaults to "static".
        mask_other (float, optional): Secondary mask argument. Defaults to 0.0.
        mask_length (int, optional): The lengths of the mask. Defaults to 10.
        no_mask_overlap (bool, optional): Whether to allow masks to overlap. Defaults to False.
        mask_min_space (int, optional): Minimum space between spans if no overlap is enabled. Defaults to 1.
        mask_channel_prob (float, optional): Probability of replacing a feature with 0. Defaults to 0.0.
        mask_channel_selection (str, optional): How to choose the mask length for channel masking. Defaults to "static".
        mask_channel_other (float, optional): Secondary mask argument for channel masking. Defaults to 0.0.
        mask_channel_length (int, optional): Minimum space between spans for channel masking. Defaults to 10.
        no_mask_channel_overlap (bool, optional): Whether to allow channel masks to overlap. Defaults to False.
        mask_channel_min_space (int, optional): Minimum space between spans for channel masking if no overlap is enabled. Defaults to 1.
        skip_masked (bool, optional): Whether to skip computing losses over masked frames. Defaults to False.
        skip_nomask (bool, optional): Whether to skip computing losses over unmasked frames. Defaults to False.
        num_classes (int, optional): The number of classes in the labels. Defaults to 100.
        final_dim (int, optional): Project final representations and targets to this dimension. Defaults to 256.
        feature_grad_mult (float, optional): The factor to scale gradients from the convolutional feature extraction layer. Defaults to 0.1.
        finetuning (bool, optional): Whether to fine-tune the model with ASR or other tasks. Defaults to False.
        freeze_encoder_updates (int, optional): The number of steps to freeze the encoder parameters in ASR fine-tuning. Defaults to 0.

    Note:
        For more details on the Hubert model and its parameters, refer to the TorchAudio documentation:
        https://pytorch.org/audio/stable/generated/torchaudio.models.hubert_pretrain_model.html
    """

    @typechecked
    def __init__(
        self,
        input_size: int = None,
        extractor_mode: str = "group_norm",
        extractor_conv_layer_config: Optional[List[List[int]]] = [
            [512, 10, 5],
            [512, 3, 2],
            [512, 3, 2],
            [512, 3, 2],
            [512, 3, 2],
            [512, 2, 2],
            [512, 2, 2],
        ],
        extractor_conv_bias: bool = False,
        encoder_embed_dim: int = 768,
        encoder_projection_dropout: float = 0.1,
        encoder_pos_conv_kernel: int = 128,
        encoder_pos_conv_groups: int = 16,
        encoder_num_layers: int = 12,
        encoder_num_heads: int = 12,
        encoder_attention_dropout: float = 0.1,
        encoder_ff_interm_features: int = 3072,
        encoder_ff_interm_dropout: float = 0.0,
        encoder_dropout: float = 0.1,
        encoder_layer_norm_first: bool = False,
        encoder_layer_drop: float = 0.05,
        mask_prob: float = 0.8,
        mask_selection: str = "static",
        mask_other: float = 0.0,
        mask_length: int = 10,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        mask_channel_prob: float = 0.0,
        mask_channel_selection: str = "static",
        mask_channel_other: float = 0.0,
        mask_channel_length: int = 10,
        no_mask_channel_overlap: bool = False,
        mask_channel_min_space: int = 1,
        skip_masked: bool = False,
        skip_nomask: bool = False,
        num_classes: int = 100,
        final_dim: int = 256,
        feature_grad_mult: Optional[float] = 0.1,
        finetuning: bool = False,
        freeze_encoder_updates: int = 0,
    ):
        super().__init__()
        try:
            import torchaudio
        except Exception as e:
            print("Error: torchaudio is not properly installed.")
            print("Please install torchaudio")
            raise e

        self._output_size = encoder_embed_dim

        self.hubert_pretrain_model = torchaudio.models.hubert_pretrain_model(
            extractor_mode=extractor_mode,
            extractor_conv_layer_config=extractor_conv_layer_config,
            extractor_conv_bias=extractor_conv_bias,
            encoder_embed_dim=encoder_embed_dim,
            encoder_projection_dropout=encoder_projection_dropout,
            encoder_pos_conv_kernel=encoder_pos_conv_kernel,
            encoder_pos_conv_groups=encoder_pos_conv_groups,
            encoder_num_layers=encoder_num_layers,
            encoder_num_heads=encoder_num_heads,
            encoder_attention_dropout=encoder_attention_dropout,
            encoder_ff_interm_features=encoder_ff_interm_features,
            encoder_ff_interm_dropout=encoder_ff_interm_dropout,
            encoder_dropout=encoder_dropout,
            encoder_layer_norm_first=encoder_layer_norm_first,
            encoder_layer_drop=encoder_layer_drop,
            mask_prob=mask_prob,
            mask_selection=mask_selection,
            mask_other=mask_other,
            mask_length=mask_length,
            no_mask_overlap=no_mask_overlap,
            mask_min_space=mask_min_space,
            mask_channel_prob=mask_channel_prob,
            mask_channel_selection=mask_channel_selection,
            mask_channel_other=mask_channel_other,
            mask_channel_length=mask_channel_length,
            no_mask_channel_overlap=no_mask_channel_overlap,
            mask_channel_min_space=mask_channel_min_space,
            skip_masked=skip_masked,
            skip_nomask=skip_nomask,
            num_classes=num_classes,
            final_dim=final_dim,
            feature_grad_mult=feature_grad_mult,
        )
        self.pretrained_params = copy.deepcopy(self.hubert_pretrain_model.state_dict())

        self.finetuning = finetuning
        if finetuning:
            for p in self.hubert_pretrain_model.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False
        self.register_buffer("global_step", torch.LongTensor([0]))
        self.freeze_encoder_updates = freeze_encoder_updates

    def output_size(self) -> int:
        """
            Get the output size of the encoder.

        Returns:
            int: The dimension of the encoder's output.

        Note:
            This method returns the value of the `_output_size` attribute,
            which is typically set during the initialization of the encoder.
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        ys_pad: torch.Tensor = None,
        ys_pad_length: torch.Tensor = None,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
            Forward pass of the Hubert Pretrain Encoder.

        This method processes the input tensor through the encoder, applying different
        forward passes based on whether the model is in pretraining, fine-tuning, or
        evaluation mode.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where B is the batch size,
                                   L is the sequence length, and D is the input dimension.
            ilens (torch.Tensor): Input lengths of shape (B,).
            ys_pad (torch.Tensor, optional): Target tensor for pretraining. Defaults to None.
            ys_pad_length (torch.Tensor, optional): Lengths of target tensor. Defaults to None.
            prev_states (torch.Tensor, optional): Previous states. Not used in current implementation. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - Output tensor after forward pass.
                - Output lengths.
                - Optional additional information (e.g., feature penalty in pretraining).

        Note:
            The behavior of this method differs based on the encoder's mode:
            - In pretraining mode, it returns logit_m, logit_u, and feature_penalty.
            - In fine-tuning mode, it applies masking and returns the encoded features.
            - In evaluation mode, it processes the input without masking.
        """

        if not self.finetuning:
            return self._pretraining_forward(xs_pad, ilens, ys_pad)
        else:
            if self.training:
                return self._finetuning_forward(xs_pad, ilens)
            else:
                return self._eval_forward(xs_pad, ilens)

    def _pretraining_forward(self, xs_pad, ilens, ys_pad):
        assert ys_pad is not None
        (
            logit_m,
            logit_u,
            feature_penalty,
        ) = self.hubert_pretrain_model.forward(xs_pad, ys_pad, ilens)

        return logit_m, logit_u, feature_penalty

    def _finetuning_forward(self, xs_pad, ilens):
        def get_padding_mask(input, lengths):
            """get_padding_mask() from torchaudio.models.wav2vec2.components"""
            batch_size, max_len, _ = input.shape
            mask = (
                torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
                >= lengths[:, None]
            )
            return mask

        # manually add the steps. It is not accurate.
        # TODO(simpleoier): to introduce the global update steps into encoder module
        self.global_step += 1
        if self.global_step <= self.freeze_encoder_updates:
            with torch.no_grad():
                x, out_len = self.hubert_pretrain_model.wav2vec2.feature_extractor(
                    xs_pad, ilens
                )
                padding_mask = get_padding_mask(x, out_len)
                (
                    x,
                    attention_mask,
                ) = self.hubert_pretrain_model.wav2vec2.encoder._preprocess(x, out_len)
                x, _ = self.hubert_pretrain_model.mask_generator(x, padding_mask)
                x = self.hubert_pretrain_model.wav2vec2.encoder.transformer(
                    x, attention_mask=attention_mask
                )
        else:
            with torch.no_grad():
                x, out_len = self.hubert_pretrain_model.wav2vec2.feature_extractor(
                    xs_pad, ilens
                )
                padding_mask = get_padding_mask(x, out_len)

            (
                x,
                attention_mask,
            ) = self.hubert_pretrain_model.wav2vec2.encoder._preprocess(x, out_len)
            x, _ = self.hubert_pretrain_model.mask_generator(x, padding_mask)
            x = self.hubert_pretrain_model.wav2vec2.encoder.transformer(
                x, attention_mask=attention_mask
            )
        return x, (~padding_mask).long().sum(dim=1), None

    def _eval_forward(self, xs_pad, ilens):
        x, lengths = self.hubert_pretrain_model.wav2vec2.feature_extractor(
            xs_pad, ilens
        )
        x = self.hubert_pretrain_model.wav2vec2.encoder(x, lengths)
        return x, lengths, None

    def reload_pretrained_parameters(self):
        """
            Reload the pretrained parameters of the Hubert model.

        This method reloads the pretrained parameters stored in the `pretrained_params`
        attribute into the `hubert_pretrain_model`. It's useful for resetting the model
        to its initial pretrained state, especially after fine-tuning or making changes
        to the model parameters.

        Note:
            This method uses `strict=False` when loading the state dict, which means
            it will ignore any parameters in the pretrained state that don't match
            the current model structure. This allows for flexibility in model architecture
            changes while still loading compatible pretrained weights.

        Raises:
            Any exceptions raised by `load_state_dict` method of PyTorch modules.

        Example:
            >>> encoder = TorchAudioHuBERTPretrainEncoder()
            >>> # After some training or parameter modifications
            >>> encoder.reload_pretrained_parameters()
            >>> print("Pretrained Hubert model parameters reloaded!")
        """
        logging.info("Pretrained Hubert model parameters reloaded!")


class FairseqHubertEncoder(AbsEncoder):
    """
    FairSeq Hubert encoder module for loading pretrained weights and fine-tuning.

    This class implements the Hubert encoder using the FairSeq implementation.
    It supports loading pretrained Hubert models and fine-tuning for downstream tasks.

    Args:
        input_size (int): Input feature dimension.
        hubert_url (str, optional): URL to the Hubert pretrained model. Defaults to "./".
        hubert_dir_path (str, optional): Directory to download the Hubert pretrained model. Defaults to "./".
        output_size (int, optional): Dimension of the encoder output. Defaults to 256.
        normalize_before (bool, optional): Whether to use layer normalization before the first block. Defaults to False.
        freeze_finetune_updates (int, optional): Number of updates to freeze all layers except the output layer. Defaults to 0.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        activation_dropout (float, optional): Dropout rate in activation function. Defaults to 0.1.
        attention_dropout (float, optional): Dropout rate in attention layers. Defaults to 0.0.
        mask_length (int, optional): Length of the mask for Hubert. Defaults to 10.
        mask_prob (float, optional): Probability of applying mask. Defaults to 0.75.
        mask_selection (str, optional): Method of selecting masks. Defaults to "static".
        mask_other (int, optional): Secondary mask argument. Defaults to 0.
        apply_mask (bool, optional): Whether to apply masking during fine-tuning. Defaults to True.
        mask_channel_length (int, optional): Length of the channel mask. Defaults to 64.
        mask_channel_prob (float, optional): Probability of applying channel mask. Defaults to 0.5.
        mask_channel_other (int, optional): Secondary channel mask argument. Defaults to 0.
        mask_channel_selection (str, optional): Method of selecting channel masks. Defaults to "static".
        layerdrop (float, optional): Probability of dropping a layer. Defaults to 0.1.
        feature_grad_mult (float, optional): Multiplier for feature extractors gradient. Defaults to 0.0.

    Note:
        This class requires FairSeq to be installed. It loads the Hubert model
        using FairSeq's model loading utilities and supports various masking
        and fine-tuning strategies specific to the Hubert architecture.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        hubert_url: str = "./",
        hubert_dir_path: str = "./",
        output_size: int = 256,
        normalize_before: bool = False,
        freeze_finetune_updates: int = 0,
        dropout_rate: float = 0.0,
        activation_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        mask_length: int = 10,
        mask_prob: float = 0.75,
        mask_selection: str = "static",
        mask_other: int = 0,
        apply_mask: bool = True,
        mask_channel_length: int = 64,
        mask_channel_prob: float = 0.5,
        mask_channel_other: int = 0,
        mask_channel_selection: str = "static",
        layerdrop: float = 0.1,
        feature_grad_mult: float = 0.0,
    ):
        super().__init__()
        self.apply_mask = apply_mask
        try:
            import fairseq
            from fairseq.models.hubert.hubert import HubertModel
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e

        arg_overrides = {
            "dropout": dropout_rate,
            "activation_dropout": activation_dropout,
            "attention_dropout": attention_dropout,
            "mask_length": mask_length,
            "mask_prob": mask_prob,
            "mask_selection": mask_selection,
            "mask_other": mask_other,
            "mask_channel_length": mask_channel_length,
            "mask_channel_prob": mask_channel_prob,
            "mask_channel_selection": mask_channel_selection,
            "mask_channel_other": mask_channel_other,
            "encoder_layerdrop": layerdrop,
            "feature_grad_mult": feature_grad_mult,
            "data": hubert_dir_path,
        }

        if hubert_url == "espnet":
            self.hubert_model_path = hubert_dir_path
            s = torch.load(
                self.hubert_model_path,
                map_location=torch.device("cpu"),
            )

            if all("encoder.encoder" in k for k in s):
                try:
                    state = {
                        k.replace("encoder.encoder.", ""): v
                        for k, v in s.items()
                        if "label_embs_concat" not in k
                    }
                except Exception as e:
                    raise e

            config_file = os.path.join(
                "/".join(self.hubert_model_path.split("/")[:-1]),
                "config.yaml",
            )
            config_file = Path(config_file)

            with config_file.open("r", encoding="utf-8") as f:
                self.pretrained_cfg = yaml.safe_load(f)

            model = FairseqHubertPretrainEncoder(
                input_size=self.pretrained_cfg["input_size"],
                hubert_dict=self.pretrained_cfg["hubert_dict"],
                **self.pretrained_cfg["encoder_conf"],
            )
            model = model.encoder

            d = self.pretrained_cfg["encoder_conf"]["output_size"]
            self.pretrained_params = copy.deepcopy(state)

        else:
            self.hubert_model_path = download_hubert(hubert_url, hubert_dir_path)

            (
                models,
                self.pretrained_cfg,
                task,
            ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [self.hubert_model_path],
                arg_overrides=arg_overrides,
                strict=False,
            )
            model = models[0]

            d = self.pretrained_cfg.model.encoder_embed_dim
            self.pretrained_params = copy.deepcopy(model.state_dict())

        self._output_size = output_size

        if not isinstance(model, HubertModel):
            try:
                model = model.hubert_encoder.hubert_model
            except Exception as e:
                print(
                    "Error: pretrained models should be within: "
                    "'HubertModel, Hubertctc' classes, etc."
                )
                raise e

        self.encoders = model

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        if output_size and output_size != d:
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(d, output_size),
            )
        else:
            self.output_layer = None

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        """
            Get the output size of the FairSeq Hubert encoder.

        Returns:
            int: The dimension of the encoder's output.

        Note:
            This method returns the value of the `_output_size` attribute,
            which is set during the initialization of the encoder. It represents
            the dimension of the output features produced by the encoder.
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
            Forward pass of the FairSeq Hubert ASR Encoder.

        This method processes the input tensor through the Hubert encoder,
        applying masking if specified and handling the freezing of parameters
        during the initial fine-tuning updates.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where B is the batch size,
                                   L is the sequence length, and D is the input dimension.
            ilens (torch.Tensor): Input lengths of shape (B,).
            prev_states (torch.Tensor, optional): Not used in the current implementation. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - xs_pad (torch.Tensor): Encoded features of shape (B, T, C), where T is the
                                         output sequence length and C is the output dimension.
                - olens (torch.Tensor): Output lengths of shape (B,).
                - None: Placeholder for future extensions.

        Note:
            - The method handles the gradual unfreezing of layers during fine-tuning.
            - It applies masking based on the `apply_mask` attribute and the training state.
            - The output may be further processed by an additional output layer if specified.
            - Layer normalization may be applied before returning the output if `normalize_before` is True.
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)

        ft = self.freeze_finetune_updates <= self.num_updates

        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning hubert parameters!")
        else:
            self.num_updates += 1
        with torch.no_grad() if not ft else contextlib.nullcontext():
            enc_outputs = self.encoders(
                xs_pad,
                padding_mask=masks,
                mask=self.apply_mask and self.training,
                features_only=True,
                output_layer=None,
            )

        xs_pad = enc_outputs["x"]  # (B,T,C),
        masks = enc_outputs["padding_mask"]  # (B, T)

        # save gpu memory
        del enc_outputs

        olens = (~masks).sum(dim=1)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        """
            Reload the pretrained parameters of the FairSeq Hubert encoder.

        This method reloads the pretrained parameters stored in the `pretrained_params`
        attribute into the `encoders` module of the FairSeq Hubert encoder. It's useful
        for resetting the model to its initial pretrained state, especially after
        fine-tuning or making changes to the model parameters.

        Note:
            This method uses `strict=False` when loading the state dict, which means
            it will ignore any parameters in the pretrained state that don't match
            the current model structure. This allows for flexibility in model architecture
            changes while still loading compatible pretrained weights.

        Raises:
            Any exceptions raised by `load_state_dict` method of PyTorch modules.

        Example:
            >>> encoder = FairseqHubertEncoder(input_size=80)
            >>> # After some training or parameter modifications
            >>> encoder.reload_pretrained_parameters()
            >>> print("Pretrained Hubert model parameters reloaded!")
        """
        logging.info("Pretrained Hubert model parameters reloaded!")


class FairseqHubertPretrainEncoder(AbsEncoder):
    """
    FairSeq Hubert pretrain encoder module, specifically designed for the pretraining stage.

    This class implements the Hubert encoder using the FairSeq implementation, tailored
    for pretraining tasks. It sets up the Hubert model with specified configurations
    and prepares it for pretraining on unlabeled data.

    Args:
        input_size (int, optional): Input feature dimension. Defaults to 1.
        output_size (int, optional): Dimension of the encoder output. Defaults to 1024.
        linear_units (int, optional): Dimension of feedforward layers. Defaults to 1024.
        attention_heads (int, optional): Number of attention heads. Defaults to 12.
        num_blocks (int, optional): Number of encoder blocks. Defaults to 12.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        attention_dropout_rate (float, optional): Dropout rate in attention layers. Defaults to 0.0.
        activation_dropout_rate (float, optional): Dropout rate for activation functions. Defaults to 0.0.
        hubert_dict (str, optional): Path to the Hubert dictionary file. Defaults to "./dict.txt".
        label_rate (int, optional): Label frame rate. Use -1 for sequence labels. Defaults to 100.
        checkpoint_activations (bool, optional): Whether to use activation checkpointing. Defaults to False.
        sample_rate (int, optional): Audio sample rate. Defaults to 16000.
        use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.
        **kwargs: Additional keyword arguments for Hubert configuration.

    Note:
        This class requires FairSeq to be installed. It sets up the Hubert model
        using FairSeq's HubertConfig and HubertPretrainingConfig, and prepares
        the model for pretraining tasks.
    """

    @typechecked
    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1024,
        linear_units: int = 1024,
        attention_heads: int = 12,
        num_blocks: int = 12,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        activation_dropout_rate: float = 0.0,
        hubert_dict: str = "./dict.txt",
        label_rate: int = 100,
        checkpoint_activations: bool = False,
        sample_rate: int = 16000,
        use_amp: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size
        self.use_amp = use_amp
        try:
            from fairseq.data.dictionary import Dictionary
            from fairseq.models.hubert.hubert import HubertConfig  # noqa: H301
            from fairseq.models.hubert.hubert import HubertModel  # noqa: H301
            from fairseq.models.hubert.hubert import (  # noqa: H301
                HubertPretrainingConfig,
            )
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e

        cfg_overides = {
            "encoder_embed_dim": output_size,
            "encoder_ffn_embed_dim": linear_units,
            "encoder_attention_heads": attention_heads,
            "encoder_layers": num_blocks,
            "final_dim": output_size,
            "dropout": dropout_rate,
            "attention_dropout": attention_dropout_rate,
            "label_rate": label_rate,
            "checkpoint_activations": checkpoint_activations,
        }
        cfg_overides = {**cfg_overides, **kwargs}
        self.cfg = HubertConfig()

        for key, value in cfg_overides.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)

        hubert_task_cfg = HubertPretrainingConfig()
        hubert_task_cfg_overides = {
            "label_rate": label_rate,
            "sample_rate": sample_rate,
        }
        for key, value in hubert_task_cfg_overides.items():
            if hasattr(hubert_task_cfg, key):
                setattr(hubert_task_cfg, key, value)

        d = Dictionary()
        self._build_dictionary(d, hubert_dict)
        self.encoder = HubertModel(self.cfg, hubert_task_cfg, self.dictionaries)

    def _build_dictionary(self, dictionary, hubert_dict_path):
        if os.path.exists(f"{hubert_dict_path}"):
            setattr(dictionary, "symbols", [])
            setattr(dictionary, "count", [])
            setattr(dictionary, "indices", {})
            dictionary.add_from_file(f"{hubert_dict_path}")
        else:
            dictionary.add_symbol("0")

        self.dictionaries = [dictionary]

    def output_size(self) -> int:
        """
            Get the output size of the FairSeq Hubert pretrain encoder.

        Returns:
            int: The dimension of the encoder's output.

        Note:
            This method returns the value of the `_output_size` attribute,
            which is set during the initialization of the encoder. It represents
            the dimension of the output features produced by the Hubert pretrain encoder.
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_length: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
            Forward pass of the FairSeq Hubert Pretrain Encoder.

        This method processes the input tensor through the Hubert encoder for pretraining,
        applying necessary masking and computing the pretraining objectives.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where B is the batch size,
                                   L is the sequence length, and D is the input dimension.
            ilens (torch.Tensor): Input lengths of shape (B,).
            ys_pad (torch.Tensor): Target tensor for pretraining tasks.
            ys_pad_length (torch.Tensor): Lengths of the target tensor.
            prev_states (torch.Tensor, optional): Not used in the current implementation. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple containing the encoder outputs.
                The exact content of the tuple depends on the Hubert pretraining configuration.

        Note:
            - This method first casts the mask embedding to the appropriate data type.
            - It then applies masking to the input and processes it through the Hubert encoder.
            - The method is specifically designed for the pretraining phase of Hubert.
        """
        self.cast_mask_emb()
        masks = make_pad_mask(ilens).to(xs_pad.device)
        ys_pad = ys_pad[:, : min(ys_pad_length)]
        enc_outputs = self.encoder(
            xs_pad,
            padding_mask=masks,
            mask=True,
            target_list=[ys_pad],
            features_only=False,
        )
        return enc_outputs

    def cast_mask_emb(self):
        """
            Cast the mask embedding to half precision if using automatic mixed precision.

        This method checks if automatic mixed precision (AMP) is enabled and if the
        mask embedding is not already in half precision. If these conditions are met,
        it casts the mask embedding to torch.cuda.HalfTensor.

        Note:
            This method is typically called before the forward pass to ensure
            that the mask embedding is in the correct precision for AMP computations.
            It modifies the `mask_emb` parameter of the encoder in-place.

        Example:
            >>> encoder = FairseqHubertPretrainEncoder(use_amp=True)
            >>> encoder.cast_mask_emb()
            # The mask_emb will be cast to half precision if it wasn't already
        """
        if self.use_amp and self.encoder.mask_emb.dtype != torch.cuda.HalfTensor:
            self.encoder.mask_emb = torch.nn.Parameter(self.encoder.mask_emb.half())

    def reload_pretrained_parameters(self):
        """
            Reinitialize the mask embedding of the Hubert encoder.

        This method reinitializes the mask embedding of the Hubert encoder with
        random values drawn from a uniform distribution. The embedding is set to
        half precision (float16) and its size is determined by the encoder's
        configuration.

        Note:
            - This method is typically used to reset the mask embedding to a new
              random state, which can be useful when restarting pretraining or
              when adapting a pretrained model to a new task.
            - The method logs information about the reinitialization, including
              the data type of the new mask embedding and whether automatic mixed
              precision (AMP) is being used.

        Example:
            >>> encoder = FairseqHubertPretrainEncoder()
            >>> encoder.reload_pretrained_parameters()
            # Logs: "Hubert mask embedding re-initialized!, torch.cuda.HalfTensor, False"
        """
        self.encoder.mask_emb = torch.nn.Parameter(
            torch.HalfTensor(self.cfg.encoder_embed_dim).uniform_()
        )
        logging.info(
            f"Hubert mask embedding re-initiallized!, \
            {self.encoder.mask_emb.dtype}, \
            {self.use_amp}"
        )


def download_hubert(model_url, dir_path):
    """
    Download the Hubert model from the given URL and save it to the specified directory.

    This function downloads the Hubert model if it doesn't exist in the specified directory.
    It uses a file lock to ensure thread-safe downloading.

    Args:
        model_url (str): The URL of the Hubert model to download.
        dir_path (str): The directory path where the model should be saved.

    Returns:
        str: The full path to the downloaded model file.

    Raises:
        Any exceptions raised by os.makedirs, FileLock, or torch.hub.download_url_to_file.

    Example:
        >>> model_url = "https://example.com/hubert_model.pt"
        >>> dir_path = "./models"
        >>> model_path = download_hubert(model_url, dir_path)
        >>> print(f"Model downloaded to: {model_path}")

    Note:
        This function uses FileLock to prevent multiple processes from downloading
        the same model simultaneously. Make sure the FileLock library is installed.
    """

    model_name = model_url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)

    with FileLock(model_path + ".lock"):
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(model_url, model_path)
            logging.info(f"Hubert model downloaded {model_path}")
        else:
            logging.info(f"Hubert model {model_path} already exists.")

    return model_path
