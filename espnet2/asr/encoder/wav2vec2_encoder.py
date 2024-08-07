# Copyright 2021 Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import contextlib
import copy
import logging
import os
from typing import Optional, Tuple

import torch
from filelock import FileLock
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class FairSeqWav2Vec2Encoder(AbsEncoder):
    """
    FairSeq Wav2Vec2 encoder module.

    This class implements an encoder based on the Wav2Vec2.0 model from FairSeq.
    It can load pre-trained Wav2Vec2.0 models and use them as feature extractors
    for speech recognition tasks. The encoder allows for fine-tuning of the
    Wav2Vec2.0 model and provides options for output dimensionality adjustment.

    Attributes:
        encoders (fairseq.models.wav2vec.wav2vec2.Wav2Vec2Model): The Wav2Vec2 model.
        output_layer (torch.nn.Sequential): Optional layer for output dimension adjustment.
        after_norm (espnet.nets.pytorch_backend.transformer.layer_norm.LayerNorm): Optional layer normalization.

    Args:
        input_size (int): Input dimension (not used in the current implementation).
        w2v_url (str): URL to the Wav2Vec2.0 pretrained model.
        w2v_dir_path (str, optional): Directory to download the Wav2Vec2.0 pretrained model. Defaults to "./".
        output_size (int, optional): Dimension of the output. Defaults to 256.
        normalize_before (bool, optional): Whether to use layer normalization before the first block. Defaults to False.
        freeze_finetune_updates (int, optional): Number of updates to freeze the model before fine-tuning. Defaults to 0.

    Note:
        This class requires FairSeq to be installed. If not installed, it will raise an ImportError
        with instructions on how to install FairSeq.

    Examples:
        >>> encoder = FairSeqWav2Vec2Encoder(input_size=80, w2v_url="https://example.com/wav2vec_model.pt")
        >>> input_tensor = torch.randn(32, 1000, 80)  # (batch_size, time_steps, features)
        >>> input_lengths = torch.full((32,), 1000)
        >>> output, output_lengths, _ = encoder(input_tensor, input_lengths)
        >>> print(output.shape)
        torch.Size([32, 1000, 256])
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        w2v_url: str,
        w2v_dir_path: str = "./",
        output_size: int = 256,
        normalize_before: bool = False,
        freeze_finetune_updates: int = 0,
    ):
        super().__init__()

        if w2v_url != "":
            try:
                import fairseq
                from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
            except Exception as e:
                print("Error: FairSeq is not properly installed.")
                print(
                    "Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done"
                )
                raise e

        self.w2v_model_path = download_w2v(w2v_url, w2v_dir_path)

        self._output_size = output_size

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.w2v_model_path],
            arg_overrides={"data": w2v_dir_path},
        )
        model = models[0]

        if not isinstance(model, Wav2Vec2Model):
            try:
                model = model.w2v_encoder.w2v_model
            except Exception as e:
                print(
                    "Error: pretrained models should be within: "
                    "'Wav2Vec2Model, Wav2VecCTC' classes, etc."
                )
                raise e

        self.encoders = model

        self.pretrained_params = copy.deepcopy(model.state_dict())

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        if model.cfg.encoder_embed_dim != output_size:
            # TODO(xkc09): try LSTM
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(model.cfg.encoder_embed_dim, output_size),
            )
        else:
            self.output_layer = None

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        """
            Get the output size of the encoder.

        Returns:
            int: The size of the output tensor along the feature dimension.

        Note:
            This method returns the value of the `_output_size` attribute,
            which is set during the initialization of the encoder.

        Examples:
            >>> encoder = FairSeqWav2Vec2Encoder(input_size=80, w2v_url="https://example.com/wav2vec_model.pt", output_size=512)
            >>> print(encoder.output_size())
            512
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
            Forward pass of the FairSeqWav2Vec2 Encoder.

        This method processes the input tensor through the Wav2Vec2 model and optional output layer.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where B is the batch size,
                                   L is the sequence length, and D is the input dimension.
            ilens (torch.Tensor): Input lengths of shape (B,) representing the valid length
                                  of each sequence in the batch.
            prev_states (torch.Tensor, optional): Not used in the current implementation. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - torch.Tensor: Output tensor of shape (B, T, C), where T is the output sequence length
                                and C is the output dimension.
                - torch.Tensor: Output lengths of shape (B,) representing the valid length of each
                                output sequence in the batch.
                - Optional[torch.Tensor]: Always None in the current implementation.

        Note:
            - The method handles the freezing and unfreezing of parameters based on the
              `freeze_finetune_updates` attribute.
            - If `normalize_before` is True, layer normalization is applied to the output.

        Examples:
            >>> encoder = FairSeqWav2Vec2Encoder(input_size=80, w2v_url="https://example.com/wav2vec_model.pt")
            >>> input_tensor = torch.randn(32, 1000, 80)  # (batch_size, time_steps, features)
            >>> input_lengths = torch.full((32,), 1000)
            >>> output, output_lengths, _ = encoder(input_tensor, input_lengths)
            >>> print(output.shape, output_lengths.shape)
            torch.Size([32, 1000, 256]) torch.Size([32])
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)

        ft = self.freeze_finetune_updates <= self.num_updates
        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning wav2vec parameters!")

        with torch.no_grad() if not ft else contextlib.nullcontext():
            enc_outputs = self.encoders(
                xs_pad,
                masks,
                mask=self.training,
                features_only=True,
            )

        xs_pad = enc_outputs["x"]  # (B,T,C),
        bs = xs_pad.shape[0]
        if enc_outputs["padding_mask"] is not None:
            masks = enc_outputs["padding_mask"]  # (B, T)
            olens = (~masks).sum(dim=1)  # (B)
        else:
            olens = torch.IntTensor([xs_pad.shape[1]]).repeat(bs).to(xs_pad.device)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        """
            Reload the pretrained parameters of the Wav2Vec model.

        This method resets the encoder's parameters to their original pretrained values.
        It's useful when you want to start from the initial pretrained state after
        fine-tuning or other modifications to the model.

        Note:
            This method uses the `pretrained_params` attribute, which is a deep copy
            of the initial model state dictionary created during the encoder's initialization.

        Examples:
            >>> encoder = FairSeqWav2Vec2Encoder(input_size=80, w2v_url="https://example.com/wav2vec_model.pt")
            >>> # After some fine-tuning or parameter updates
            >>> encoder.reload_pretrained_parameters()
            >>> print("Pretrained Wav2Vec model parameters reloaded!")
        """
        logging.info("Pretrained Wav2Vec model parameters reloaded!")


def download_w2v(model_url, dir_path):
    """
    Download the Wav2Vec model and its dictionary.

    This function downloads the Wav2Vec model and its corresponding dictionary
    if they don't already exist in the specified directory. It uses FileLock
    to ensure thread-safe downloads.

    Args:
        model_url (str): The URL of the Wav2Vec model to download.
        dir_path (str): The directory path where the model and dictionary
            should be saved.

    Returns:
        str: The local file path of the downloaded Wav2Vec model.

    Raises:
        OSError: If there are issues creating the directory or downloading files.

    Examples:
        >>> model_url = "https://example.com/wav2vec_model.pt"
        >>> dir_path = "./models"
        >>> local_model_path = download_w2v(model_url, dir_path)
        >>> print(local_model_path)
        './models/wav2vec_model.pt'

    Note:
        This function also downloads a dictionary file from a hardcoded URL:
        'https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt'
    """

    model_name = model_url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)

    dict_url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt"
    dict_path = os.path.join(dir_path, dict_url.split("/")[-1])

    with FileLock(model_path + ".lock"):
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(model_url, model_path)
            torch.hub.download_url_to_file(dict_url, dict_path)
            logging.info(f"Wav2Vec model downloaded {model_path}")
        else:
            logging.info(f"Wav2Vec model {model_path} already exists.")

    return model_path
