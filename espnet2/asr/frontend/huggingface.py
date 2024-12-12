import copy
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
import torch.share


class HuggingFaceFrontend(AbsFrontend):
    """Use pretrained models from Hugging Face Transformers for ASR"""

    @typechecked
    def __init__(
        self,
        model,
        fs: Union[int, str] = 16000,
        download_dir: Optional[str] = None,
    ):
        try:
            from transformers import (
                AutoFeatureExtractor,
                AutoModel,
                EncodecFeatureExtractor,
                WhisperFeatureExtractor,
            )
        except ImportError:
            raise ImportError("Please install `transformers`")

        super().__init__()
        self.encoder = AutoModel.from_pretrained(model, cache_dir=download_dir)
        self.processor = AutoFeatureExtractor.from_pretrained(
            model, cache_dir=download_dir
        )
        if isinstance(self.processor, EncodecFeatureExtractor) or isinstance(
            self.processor, WhisperFeatureExtractor
        ):
            raise ValueError("Frontend not supported.")
        self.pretrained_params = copy.deepcopy(self.encoder.state_dict())

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != self.processor.sampling_rate:
            raise ValueError(
                f"Specified sampling rate {fs} does not match that of "
                f"the pretrained model: {self.processor.sampling_rate}."
            )

    def output_size(self) -> int:
        return self.encoder.config.hidden_size

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs will follow ESPNet conventions which are padded tensors,
        but we need to re-convert to numpy and re-encode with the HF processor.
        """
        with torch.no_grad():
            # Re-obtain jagged inputs to feed into the HF processor
            device = inputs.device
            inputs = [arr[:l].cpu().numpy() for arr, l in zip(inputs, input_lengths)]
            encoded = self.processor(
                inputs,
                return_tensors="pt",
                sampling_rate=self.processor.sampling_rate,
                padding=True,
            ).to(device)
            if "attention_mask" not in encoded:
                encoded_lengths = torch.tensor(encoded.input_values.shape)
            else:
                encoded_lengths = torch.sum(encoded.attention_mask, dim=-1)

        encoded = self.encoder(**encoded).last_hidden_state
        if torch.max(encoded_lengths) != encoded.size(1):
            # truncate the sequence to the actual length
            # there is a weird bug in conformer encoder
            encoded = encoded[:, : torch.max(encoded_lengths), :]

        return encoded, encoded_lengths

    def reload_pretrained_parameters(self):
        self.encoder.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Transformers model parameters reloaded!")
