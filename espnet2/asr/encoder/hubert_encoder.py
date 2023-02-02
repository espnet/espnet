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
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class TorchAudioHuBERTPretrainEncoder(AbsEncoder):
    """Torch Audio Hubert encoder module.

    Args:
        extractor_mode: Operation mode of feature extractor.
            Valid values are "group_norm" or "layer_norm".
        extractor_conv_layer_config: Configuration of convolution layers in feature
            extractor. List of convolution configuration,
            i.e. [(output_channel, kernel_size, stride), ...]
        extractor_conv_bias: Whether to include bias term to each convolution
            operation.
        encoder_embed_dim: The dimension of embedding in encoder.
        encoder_projection_dropout: The dropout probability applied after the input
            feature is projected to "encoder_embed_dim".
        encoder_pos_conv_kernel: Kernel size of convolutional positional embeddings.
        encoder_pos_conv_groups: Number of groups of convolutional positional
            embeddings.
        encoder_num_layers: Number of self attention layers in transformer block.
        encoder_num_heads: Number of heads in self attention layers.
        encoder_attention_dropout: Dropout probability applied after softmax in
            self-attention layer.
        encoder_ff_interm_features: Dimension of hidden features in feed forward layer.
        encoder_ff_interm_dropout: Dropout probability applied in feedforward layer.
        encoder_dropout: Dropout probability applied at the end of feed forward layer.
        encoder_layer_norm_first: Control the order of layer norm in transformer layer
            and each encoder layer. If True, in transformer layer, layer norm is
            applied before features are fed to encoder layers.
        encoder_layer_drop: Probability to drop each encoder layer during training.
        mask_prob: Probability for each token to be chosen as start of the span
            to be masked.
        mask_selection: How to choose the mask length.
            Options: [static, uniform, normal, poisson].
        mask_other: Secondary mask argument (used for more complex distributions).
        mask_length: The lengths of the mask.
        no_mask_overlap: Whether to allow masks to overlap.
        mask_min_space: Minimum space between spans (if no overlap is enabled).
        mask_channel_prob: (float): The probability of replacing a feature with 0.
        mask_channel_selection: How to choose the mask length for channel masking.
            Options: [static, uniform, normal, poisson].
        mask_channel_other: Secondary mask argument for channel masking(used for more
            complex distributions).
        mask_channel_length: Minimum space between spans (if no overlap is enabled)
            for channel masking.
        no_mask_channel_overlap: Whether to allow channel masks to overlap.
        mask_channel_min_space: Minimum space between spans for channel
            masking(if no overlap is enabled).
        skip_masked: If True, skip computing losses over masked frames.
        skip_nomask: If True, skip computing losses over unmasked frames.
        num_classes: The number of classes in the labels.
        final_dim: Project final representations and targets to final_dim.
        feature_grad_mult: The factor to scale the convolutional feature extraction
            layer gradients by. The scale factor will not affect the forward pass.
        finetuning: Whether to finetuning the model with ASR or other tasks.
        freeze_encoder_updates: The number of steps to freeze the encoder parameters
            in ASR finetuning.
    Hubert specific Args:
        Please refer to:
        https://pytorch.org/audio/stable/generated/torchaudio.models.hubert_pretrain_model.html#torchaudio.models.hubert_pretrain_model
    """

    def __init__(
        self,
        input_size: int = None,
        extractor_mode: str = "group_norm",
        extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]] = [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
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
        assert check_argument_types()
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
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        ys_pad: torch.Tensor = None,
        ys_pad_length: torch.Tensor = None,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward Hubert Pretrain Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
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
        self.hubert_pretrain_model.load_state_dict(self.pretrained_params, strict=False)
        logging.info("Pretrained Hubert model parameters reloaded!")


class FairseqHubertEncoder(AbsEncoder):
    """FairSeq Hubert encoder module, used for loading pretrained weight and finetuning

    Args:
        input_size: input dim
        hubert_url: url to Hubert pretrained model
        hubert_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        output_size: dimension of attention
        normalize_before: whether to use layer_norm before the first block
        freeze_finetune_updates: steps that freeze all layers except output layer
            before tuning the whole model (nessasary to prevent overfit).
        dropout_rate: dropout rate
        activation_dropout: dropout rate in activation function
        attention_dropout: dropout rate in attention
    Hubert specific Args:
        Please refer to:
        https://github.com/pytorch/fairseq/blob/master/fairseq/models/hubert/hubert.py
    """

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
        assert check_argument_types()
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
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward Hubert ASR Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
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
        self.encoders.load_state_dict(self.pretrained_params, strict=False)
        logging.info("Pretrained Hubert model parameters reloaded!")


class FairseqHubertPretrainEncoder(AbsEncoder):
    """FairSeq Hubert pretrain encoder module, only used for pretraining stage

    Args:
        input_size: input dim
        output_size: dimension of attention
        linear_units: dimension of feedforward layers
        attention_heads: the number of heads of multi head attention
        num_blocks: the number of encoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        hubert_dict: target dictionary for Hubert pretraining
        label_rate: label frame rate. -1 for sequence label
        sample_rate: target sample rate.
        use_amp: whether to use automatic mixed precision
        normalize_before: whether to use layer_norm before the first block
    """

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
        assert check_argument_types()
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
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_length: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward Hubert Pretrain Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
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
        if self.use_amp and self.encoder.mask_emb.dtype != torch.cuda.HalfTensor:
            self.encoder.mask_emb = torch.nn.Parameter(self.encoder.mask_emb.half())

    def reload_pretrained_parameters(self):
        self.encoder.mask_emb = torch.nn.Parameter(
            torch.HalfTensor(self.cfg.encoder_embed_dim).uniform_()
        )
        logging.info(
            f"Hubert mask embedding re-initiallized!, \
            {self.encoder.mask_emb.dtype}, \
            {self.use_amp}"
        )


def download_hubert(model_url, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    model_name = model_url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)

    with FileLock(model_path + ".lock"):
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(model_url, model_path)
            logging.info(f"Hubert model downloaded {model_path}")
        else:
            logging.info(f"Hubert model {model_path} already exists.")

    return model_path
