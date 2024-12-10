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
    Torch Audio HuBERT encoder module for speech representation learning.

    This class implements the HuBERT encoder, a model designed for
    self-supervised speech representation learning. The encoder uses
    convolutional layers followed by transformer blocks, allowing for
    efficient processing of audio input.

    Args:
        extractor_mode (str): Operation mode of the feature extractor.
            Valid values are "group_norm" or "layer_norm".
        extractor_conv_layer_config (List[List[int]]): Configuration of
            convolution layers in feature extractor. List of
            convolution configurations, i.e. [[output_channel, kernel_size, stride], ...].
        extractor_conv_bias (bool): Whether to include a bias term for each
            convolution operation.
        encoder_embed_dim (int): The dimension of embedding in the encoder.
        encoder_projection_dropout (float): The dropout probability applied
            after projecting the input feature to "encoder_embed_dim".
        encoder_pos_conv_kernel (int): Kernel size of convolutional positional embeddings.
        encoder_pos_conv_groups (int): Number of groups for convolutional
            positional embeddings.
        encoder_num_layers (int): Number of self-attention layers in the transformer block.
        encoder_num_heads (int): Number of heads in the self-attention layers.
        encoder_attention_dropout (float): Dropout probability applied after
            softmax in the self-attention layer.
        encoder_ff_interm_features (int): Dimension of hidden features in the
            feedforward layer.
        encoder_ff_interm_dropout (float): Dropout probability applied in the
            feedforward layer.
        encoder_dropout (float): Dropout probability applied at the end of the
            feedforward layer.
        encoder_layer_norm_first (bool): Control the order of layer norm in
            the transformer layer and each encoder layer. If True, layer norm
            is applied before features are fed to encoder layers.
        encoder_layer_drop (float): Probability to drop each encoder layer during training.
        mask_prob (float): Probability for each token to be chosen as the start
            of the span to be masked.
        mask_selection (str): Method for choosing the mask length. Options:
            [static, uniform, normal, poisson].
        mask_other (float): Secondary mask argument for more complex distributions.
        mask_length (int): Lengths of the mask.
        no_mask_overlap (bool): Whether to allow masks to overlap.
        mask_min_space (int): Minimum space between spans if no overlap is enabled.
        mask_channel_prob (float): Probability of replacing a feature with 0.
        mask_channel_selection (str): Method for choosing the mask length for
            channel masking. Options: [static, uniform, normal, poisson].
        mask_channel_other (float): Secondary mask argument for channel masking.
        mask_channel_length (int): Minimum space between spans for channel
            masking if no overlap is enabled.
        no_mask_channel_overlap (bool): Whether to allow channel masks to overlap.
        mask_channel_min_space (int): Minimum space between spans for channel
            masking if no overlap is enabled.
        skip_masked (bool): If True, skip computing losses over masked frames.
        skip_nomask (bool): If True, skip computing losses over unmasked frames.
        num_classes (int): The number of classes in the labels.
        final_dim (int): Dimension to project final representations and targets.
        feature_grad_mult (Optional[float]): Factor to scale the convolutional
            feature extraction layer gradients. The scale factor does not affect
            the forward pass.
        finetuning (bool): Whether to fine-tune the model with ASR or other tasks.
        freeze_encoder_updates (int): Number of steps to freeze the encoder
            parameters in ASR fine-tuning.

    Hubert specific Args:
        Please refer to:
        https://pytorch.org/audio/stable/generated/torchaudio.models.hubert_pretrain_model.html#torchaudio.models.hubert_pretrain_model

    Examples:
        >>> encoder = TorchAudioHuBERTPretrainEncoder()
        >>> input_tensor = torch.randn(10, 16000)  # Example input
        >>> output = encoder(input_tensor)  # Forward pass

    Note:
        Ensure that torchaudio is installed and properly configured for
        using the HuBERT model.

    Raises:
        ImportError: If torchaudio is not installed.
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

        This method returns the dimension of the output
        from the encoder, which corresponds to the
        embedding dimension used in the model.

        Returns:
            int: The output size (dimension of the encoder
            output).

        Examples:
            >>> encoder = TorchAudioHuBERTPretrainEncoder()
            >>> encoder.output_size()
            768
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
        Forward pass for the Hubert Pretrain Encoder.

        This method processes the input tensor through the Hubert pretraining
        model. It handles both pretraining and fine-tuning modes, depending on
        the state of the model.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where B is
                the batch size, L is the sequence length, and D is the feature
                dimension.
            ilens (torch.Tensor): A tensor of shape (B,) representing the
                lengths of each input sequence in the batch.
            ys_pad (torch.Tensor, optional): Target tensor of shape (B, L_y, D)
                for training. Defaults to None.
            ys_pad_length (torch.Tensor, optional): A tensor of shape (B,)
                representing the lengths of each target sequence. Defaults to None.
            prev_states (torch.Tensor, optional): Not used in the current version.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple
            containing:
                - The position-embedded output tensor.
                - A tensor representing the mask.
                - An optional tensor, which is currently None.

        Examples:
            >>> encoder = TorchAudioHuBERTPretrainEncoder()
            >>> input_tensor = torch.randn(4, 100, 768)  # (B, L, D)
            >>> input_lengths = torch.tensor([100, 90, 80, 70])  # Lengths of each sequence
            >>> output, mask, _ = encoder.forward(input_tensor, input_lengths)

        Note:
            - If the model is in fine-tuning mode, the method will skip certain
              computations based on the specified flags.
            - Ensure that the input tensor and its lengths are correctly formatted
              to avoid runtime errors.
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
        Reloads the pretrained parameters into the Hubert model.

        This method loads the previously stored state dictionary of the
        pretrained Hubert model from `self.pretrained_params` and applies it to
        the current instance of the Hubert model. This is particularly useful
        when the model has been fine-tuned or modified and you want to revert
        to the original pretrained weights.

        Note:
            The loading is done with `strict=False`, which means that if there
            are keys in the state dict that are not found in the model, they will
            be ignored.

        Examples:
            >>> encoder = TorchAudioHuBERTPretrainEncoder()
            >>> # Assume some fine-tuning has happened here
            >>> encoder.reload_pretrained_parameters()  # Reloads original weights

        Raises:
            RuntimeError: If the model fails to load the state dict.
        """
        logging.info("Pretrained Hubert model parameters reloaded!")


class FairseqHubertEncoder(AbsEncoder):
    """
    FairSeq Hubert encoder module for loading pretrained weights and fine-tuning.

    This class provides an interface for using the Hubert encoder architecture
    from FairSeq, enabling loading of pretrained models and fine-tuning for
    various tasks, particularly Automatic Speech Recognition (ASR).

    Args:
        input_size (int): Input dimension for the model.
        hubert_url (str): URL to download the Hubert pretrained model.
        hubert_dir_path (str): Directory to save the downloaded model.
        output_size (int): Dimension of the output from the encoder.
        normalize_before (bool): Whether to apply layer normalization before
            the first block of the encoder.
        freeze_finetune_updates (int): Number of updates to freeze all layers
            except the output layer before tuning the entire model.
            Necessary to prevent overfitting.
        dropout_rate (float): Dropout rate applied in the encoder.
        activation_dropout (float): Dropout rate applied in activation functions.
        attention_dropout (float): Dropout rate applied in the attention
            mechanism.

    Hubert-specific Args:
        For more details, please refer to:
        https://github.com/pytorch/fairseq/blob/master/fairseq/models/hubert/hubert.py

    Examples:
        >>> encoder = FairseqHubertEncoder(input_size=512, hubert_url="path/to/model")
        >>> xs_pad = torch.randn(10, 100, 512)  # (B, L, D)
        >>> ilens = torch.tensor([100] * 10)  # Lengths of input sequences
        >>> outputs, olens, _ = encoder(xs_pad, ilens)

    Note:
        Ensure that the FairSeq library is installed and properly configured
        to use this module.
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
        Returns the output size of the encoder.

        This method provides the dimensionality of the output from the encoder,
        which is crucial for understanding the model's representation capacity.

        Returns:
            int: The output size of the encoder, typically representing the
            dimensionality of the final layer.

        Examples:
            >>> encoder = FairseqHubertEncoder(input_size=256, output_size=512)
            >>> encoder.output_size()
            512

        Note:
            The output size is set during the initialization of the encoder
            and can be modified if necessary.
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the Hubert ASR Encoder.

        This method processes the input tensor through the Hubert encoder,
        returning the position-embedded tensor along with a mask. The behavior
        of the method depends on whether the model is in finetuning mode or
        not.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D) where B is
                the batch size, L is the sequence length, and D is the
                feature dimension.
            ilens (torch.Tensor): A tensor containing the lengths of the input
                sequences of shape (B).
            ys_pad (torch.Tensor, optional): Target tensor of shape (B, T, C),
                where T is the target sequence length and C is the number of
                classes. Default is None.
            ys_pad_length (torch.Tensor, optional): Lengths of the target
                sequences. Default is None.
            prev_states (torch.Tensor, optional): Placeholder for previous
                states. Not used in the current implementation. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple
            containing:
                - Position embedded tensor of shape (B, T, D).
                - A tensor containing the lengths of the outputs of shape (B).
                - An optional tensor (currently None).

        Note:
            The method checks if the model is in finetuning mode. If it is not,
            it calls the `_pretraining_forward` method. If the model is in
            training mode, it calls `_finetuning_forward`, and otherwise, it
            calls `_eval_forward`.

        Examples:
            >>> encoder = FairseqHubertEncoder(input_size=256)
            >>> xs_pad = torch.randn(2, 10, 256)  # Example input tensor
            >>> ilens = torch.tensor([10, 8])  # Lengths of input sequences
            >>> outputs = encoder(xs_pad, ilens)
            >>> print(outputs[0].shape)  # Position embedded tensor shape
            >>> print(outputs[1])  # Output lengths

        Raises:
            ValueError: If `ys_pad` is None when not in finetuning mode and
            `_pretraining_forward` is called.
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
        Reloads the pretrained parameters into the Hubert model.

        This method loads the parameters stored in `self.pretrained_params`
        back into the Hubert pretraining model. This can be useful when you
        want to reset the model to its initial state after fine-tuning or
        experimentation.

        It performs the following steps:
            1. Loads the state dictionary from `self.pretrained_params`.
            2. Logs a message indicating that the pretrained parameters
            have been reloaded.

        Examples:
            >>> encoder = FairseqHubertEncoder(input_size=128)
            >>> encoder.reload_pretrained_parameters()
            Pretrained Hubert model parameters reloaded!

        Note:
            The `strict` argument is set to `False` to allow loading of
            parameters that may not match exactly, which can be useful
            if some layers were added or modified.

        Raises:
            RuntimeError: If the state dictionary cannot be loaded
            due to a mismatch in parameters.
        """
        logging.info("Pretrained Hubert model parameters reloaded!")


class FairseqHubertPretrainEncoder(AbsEncoder):
    """
    FairSeq Hubert pretrain encoder module, used for the pretraining stage.

    This class implements the pretraining encoder for the Hubert model, which is
    used to learn representations of audio data through masked prediction tasks.
    It is part of the FairSeq implementation of Hubert, allowing for various
    configurations including dropout rates, sample rates, and dictionary paths.

    Attributes:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention.
        linear_units (int): Dimension of feedforward layers.
        attention_heads (int): Number of heads in multi-head attention.
        num_blocks (int): Number of encoder blocks.
        dropout_rate (float): Dropout rate applied to layers.
        attention_dropout_rate (float): Dropout rate applied to attention layers.
        hubert_dict (str): Path to the target dictionary for Hubert pretraining.
        label_rate (int): Frame rate for labels. -1 indicates sequence labels.
        sample_rate (int): Target sample rate for audio data.
        use_amp (bool): Indicates whether to use automatic mixed precision.

    Args:
        input_size: Input dimension.
        output_size: Dimension of attention.
        linear_units: Dimension of feedforward layers.
        attention_heads: Number of heads in multi-head attention.
        num_blocks: Number of encoder blocks.
        dropout_rate: Dropout rate for layers.
        attention_dropout_rate: Dropout rate for attention layers.
        hubert_dict: Path to the target dictionary for Hubert pretraining.
        label_rate: Frame rate for labels. -1 indicates sequence labels.
        sample_rate: Target sample rate for audio data.
        use_amp: Whether to use automatic mixed precision.

    Examples:
        # Instantiate the encoder
        encoder = FairseqHubertPretrainEncoder(
            input_size=1,
            output_size=1024,
            linear_units=1024,
            attention_heads=12,
            num_blocks=12,
            dropout_rate=0.1,
            hubert_dict="./dict.txt"
        )

        # Forward pass with input tensor
        xs_pad = torch.randn(32, 100, 1)  # (B, L, D)
        ilens = torch.tensor([100] * 32)  # Lengths
        ys_pad = torch.randn(32, 50, 1024)  # Target labels
        ys_pad_length = torch.tensor([50] * 32)  # Lengths of targets
        outputs = encoder(xs_pad, ilens, ys_pad, ys_pad_length)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            Output tensor, output lengths, and optional additional information.

    Note:
        Ensure that the required libraries (FairSeq) are properly installed to
        utilize this encoder. Check the documentation for detailed instructions
        on installation and setup.

    Todo:
        - Implement additional features for enhanced flexibility and
        performance.
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
        Get the output size of the encoder.

        This method returns the dimension of the output from the encoder,
        which corresponds to the embedding dimension used in the model.

        Returns:
            int: The output size (embedding dimension) of the encoder.

        Examples:
            >>> encoder = FairseqHubertEncoder(output_size=256)
            >>> encoder.output_size()
            256

            >>> encoder = FairseqHubertEncoder(output_size=512)
            >>> encoder.output_size()
            512
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
        Forward pass for the Hubert Pretrain Encoder.

        This method processes the input tensor through the Hubert encoder.
        Depending on whether the model is in finetuning or pretraining mode,
        it directs the input to the appropriate forward function.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where
                B is the batch size, L is the sequence length, and D is
                the feature dimension.
            ilens (torch.Tensor): Input lengths tensor of shape (B),
                containing the actual lengths of the input sequences.
            ys_pad (torch.Tensor, optional): Target tensor of shape
                (B, L_y, D), where L_y is the length of the target sequences.
                This is only used in pretraining mode. Default is None.
            ys_pad_length (torch.Tensor, optional): Lengths of the target
                sequences, used only in pretraining mode. Default is None.
            prev_states (torch.Tensor, optional): Placeholder for previous
                states, not utilized in the current implementation. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                A tuple containing:
                - position embedded tensor (B, T, D)
                - mask tensor indicating the number of valid elements in the
                  output (B)
                - Optional tensor for additional outputs, currently None.

        Examples:
            >>> encoder = TorchAudioHuBERTPretrainEncoder()
            >>> xs_pad = torch.randn(2, 100, 768)  # Example input
            >>> ilens = torch.tensor([100, 80])  # Input lengths
            >>> ys_pad = torch.randint(0, 10, (2, 50))  # Example targets
            >>> ys_pad_length = torch.tensor([50, 50])  # Target lengths
            >>> output = encoder.forward(xs_pad, ilens, ys_pad, ys_pad_length)

        Note:
            The method will return different outputs based on the finetuning
            state of the model.
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
        Cast the mask embedding to half precision if using AMP.

        This method checks if automatic mixed precision (AMP) is enabled and
        casts the mask embedding parameter of the encoder to half precision
        (float16) if it is not already in that format. This is particularly
        useful for improving performance and reducing memory usage during
        training on compatible hardware.

        Note:
            This method is typically called during the forward pass of the
            encoder to ensure that the mask embedding is in the correct
            format for mixed precision training.

        Raises:
            TypeError: If the encoder's mask embedding is not a torch
                Parameter.

        Examples:
            If `self.use_amp` is True and the mask embedding is not in
            half precision, this method will convert it:

            >>> encoder = FairseqHubertPretrainEncoder(...)
            >>> encoder.use_amp = True
            >>> encoder.cast_mask_emb()  # Mask embedding is cast to half precision.
        """
        if self.use_amp and self.encoder.mask_emb.dtype != torch.cuda.HalfTensor:
            self.encoder.mask_emb = torch.nn.Parameter(self.encoder.mask_emb.half())

    def reload_pretrained_parameters(self):
        """
        Reload the pretrained parameters for the Hubert model.

        This method loads the parameters from the previously stored state
        dictionary `pretrained_params` back into the Hubert model. It allows
        for restoring the model's weights to a pretrained state, which is useful
        during fine-tuning or after training sessions.

        Logging is performed to indicate the successful reloading of parameters.

        Examples:
            >>> encoder = FairseqHubertEncoder(...)
            >>> encoder.reload_pretrained_parameters()
            Pretrained Hubert model parameters reloaded!

        Note:
            The method uses `strict=False` when loading the state dictionary,
            which means that it will ignore any keys that are not found in the
            current model. This is particularly useful if the model architecture
            has changed since the parameters were saved.

        Raises:
            RuntimeError: If there is an issue with loading the state dictionary.
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
    Download the HuBERT model from a given URL and save it to a specified directory.

    This function checks if the model already exists in the specified directory. If
    it does not exist, it downloads the model from the provided URL and saves it
    to the specified path, ensuring that the directory structure is created if
    necessary.

    Args:
        model_url (str): The URL from which to download the HuBERT model.
        dir_path (str): The directory where the model should be saved.

    Returns:
        str: The path to the downloaded model file.

    Raises:
        Exception: If the download fails or if the specified directory cannot be
        created.

    Examples:
        >>> model_path = download_hubert(
        ...     "https://example.com/hubert_model.pt",
        ...     "./models/hubert"
        ... )
        >>> print(model_path)
        ./models/hubert/hubert_model.pt

    Note:
        The function uses a file lock to prevent concurrent downloads of the same
        model.

    Todo:
        - Add support for different model versions.
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
