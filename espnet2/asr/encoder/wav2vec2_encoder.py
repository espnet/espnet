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
    FairSeq Wav2Vec2 encoder module for automatic speech recognition.

    This encoder utilizes a pre-trained Wav2Vec2.0 model from FairSeq to extract
    features from audio input. It can be fine-tuned on specific tasks, allowing
    for flexible and powerful speech recognition capabilities.

    Args:
        input_size (int): Input dimension for the encoder.
        w2v_url (str): URL to the Wav2Vec2.0 pretrained model.
        w2v_dir_path (str, optional): Directory to download the Wav2Vec2.0
            pretrained model. Defaults to "./".
        output_size (int, optional): Dimension of the output features after
            encoding. Defaults to 256.
        normalize_before (bool, optional): Whether to apply layer normalization
            before the first block. Defaults to False.
        freeze_finetune_updates (int, optional): Number of updates after which
            the encoder parameters can be fine-tuned. Defaults to 0.

    Attributes:
        encoders: The loaded Wav2Vec2 model used for encoding.
        pretrained_params: A copy of the pretrained model's parameters for
            reloading.
        output_layer: An optional linear layer to adjust output dimensions.
        normalize_before: A flag indicating whether normalization is applied
            before encoding.
        freeze_finetune_updates: The threshold for starting fine-tuning.

    Returns:
        output_size (int): The size of the output features.

    Examples:
        >>> encoder = FairSeqWav2Vec2Encoder(
        ...     input_size=161,
        ...     w2v_url="https://path/to/wav2vec2/model",
        ...     output_size=256,
        ... )
        >>> xs_pad = torch.randn(10, 100, 161)  # (B, L, D)
        >>> ilens = torch.tensor([100] * 10)  # Input lengths
        >>> output, olens, _ = encoder(xs_pad, ilens)

    Note:
        Ensure that the FairSeq library is installed properly. You can
        install it using the command:
        `cd ${MAIN_ROOT}/tools && make fairseq.done`.

    Raises:
        Exception: If the FairSeq library is not installed or if the model
        class is not compatible.

    Todo:
        - Explore the option to implement an LSTM for output layer adjustment
          if the dimensions do not match.
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

        This method returns the dimension of the output produced by the
        encoder. It is particularly useful for understanding the shape of
        the features that will be passed to subsequent layers in a neural
        network.

        Returns:
            int: The output size of the encoder.

        Examples:
            encoder = FairSeqWav2Vec2Encoder(input_size=512, output_size=256)
            size = encoder.output_size()
            print(size)  # Output: 256
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the FairSeq Wav2Vec2 Encoder.

        This method takes padded input tensors and their lengths, processes them
        through the Wav2Vec2 encoder, and returns the encoded output along with
        the corresponding output lengths and an optional tensor.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where B is
                the batch size, L is the sequence length, and D is the feature
                dimension.
            ilens (torch.Tensor): Tensor of shape (B,) representing the lengths
                of each input sequence in the batch.
            prev_states (torch.Tensor, optional): Previous states (not used in
                this implementation). Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - A tensor containing the position-embedded output of shape
                  (B, T, C), where T is the output sequence length and C is the
                  output dimension.
                - A tensor of shape (B,) representing the lengths of the output
                  sequences.
                - An optional tensor (currently None).

        Examples:
            >>> encoder = FairSeqWav2Vec2Encoder(input_size=128, w2v_url="url")
            >>> xs_pad = torch.randn(2, 100, 128)  # Example input
            >>> ilens = torch.tensor([100, 80])  # Example lengths
            >>> output, olens, _ = encoder.forward(xs_pad, ilens)
            >>> print(output.shape)  # Should print: torch.Size([2, T, output_size])
            >>> print(olens)  # Should print the lengths of the output sequences

        Note:
            The method automatically handles fine-tuning of the encoder
            parameters based on the number of updates.

        Raises:
            RuntimeError: If the encoder fails to process the input due to
            incompatible dimensions or other issues.

        Todo:
            Implement usage of `prev_states` in future versions if needed.
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
        Reload the pretrained parameters into the encoder.

        This method loads the parameters that were initially stored in the
        `pretrained_params` attribute back into the encoder model. This is useful
        for restoring the original state of the model after fine-tuning or
        modifications have been made.

        Examples:
            # Create an instance of the encoder
            encoder = FairSeqWav2Vec2Encoder(
                input_size=256,
                w2v_url='https://example.com/wav2vec2_model',
                w2v_dir_path='./models',
            )

            # Fine-tune the encoder (hypothetical fine-tuning code here)
            # ...

            # Reload the original pretrained parameters
            encoder.reload_pretrained_parameters()

        Note:
            This operation does not return any values but logs a message
            indicating that the parameters have been reloaded.
        """
        logging.info("Pretrained Wav2Vec model parameters reloaded!")


def download_w2v(model_url, dir_path):
    """
    Download the Wav2Vec2.0 pretrained model and its dictionary file.

    This function downloads the specified Wav2Vec2.0 model from a given URL
    and saves it to the specified directory. It also downloads the associated
    dictionary file necessary for the model's operation.

    Args:
        model_url (str): The URL to the Wav2Vec2.0 pretrained model.
        dir_path (str): The directory path where the model will be downloaded.

    Returns:
        str: The path to the downloaded Wav2Vec2.0 model file.

    Raises:
        Exception: If there is an issue with downloading the model or creating
        directories.

    Examples:
        >>> model_url = "https://path/to/wav2vec2/model"
        >>> dir_path = "./models"
        >>> model_path = download_w2v(model_url, dir_path)
        >>> print(model_path)
        ./models/model

    Note:
        This function uses a file lock to prevent concurrent downloads of the
        same model.
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
