import copy
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import librosa
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend


class HuggingFaceFrontend(AbsFrontend):
    """Use pretrained models from HuggingFace for ASR"""

    @typechecked
    def __init__(
        self,
        model,
        fs: Union[int, str] = 16000,
        download_dir: Optional[str] = None,
    ):
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError:
            raise ImportError("Please install `transformers`")

        super().__init__()
        self.encoder = AutoModel.from_pretrained(model, cache_dir=download_dir)
        self.processor = AutoFeatureExtractor.from_pretrained(
            model, cache_dir=download_dir
        )
        self.pretrained_params = copy.deepcopy(self.encoder.state_dict())

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != self.processor.sampling_rate:
            logging.warning(
                f"Sampling rate {fs} does not match upstream model: "
                + str(self.processor.sampling_rate)
                + ". Resampling will be performed at forward time."
            )
            self.resample = True
            self.fs = fs
        else:
            self.resample = False

    def output_size(self) -> int:
        return self.encoder.config.hidden_size

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # Reobtain jagged inputs to feed into the HF processor
            device = inputs.device
            inputs = [arr[:l].cpu().numpy() for arr, l in zip(inputs, input_lengths)]
            if self.resample:
                inputs = [
                    librosa.resample(
                        arr, orig_sr=self.fs, target_sr=self.processor.sampling_rate
                    ).ravel()
                    for arr in inputs
                ]
            encoded = self.processor(
                inputs,
                return_tensors="pt",
                sampling_rate=self.processor.sampling_rate,
                padding=True,
            ).to(device)
            encoded_lengths = torch.sum(encoded.attention_mask, dim=-1)

        encoded = self.encoder(**encoded).last_hidden_state

        return encoded, encoded_lengths

    def reload_pretrained_parameters(self):
        self.encoder.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Transformers model parameters reloaded!")
