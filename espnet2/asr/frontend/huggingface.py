import copy
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import numpy as np
import torch
import torch.share
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend


class HuggingFaceFrontend(AbsFrontend):
    """Use pretrained models from Hugging Face Transformers for ASR"""

    @typechecked
    def __init__(
        self,
        model,
        fs: Union[int, str] = 16000,
        download_dir: Optional[str] = None,
        load_pretrained: bool = True,
    ):
        try:
            from transformers import (
                AutoConfig,
                AutoFeatureExtractor,
                AutoModel,
                EncodecFeatureExtractor,
                WhisperFeatureExtractor,
            )
        except ImportError:
            raise ImportError("Please install `transformers`")

        super().__init__()
        if load_pretrained:
            self.encoder = AutoModel.from_pretrained(model, cache_dir=download_dir)
        else:
            config = AutoConfig.from_pretrained(model, cache_dir=download_dir)
            self.encoder = AutoModel.from_config(config)
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

        # What the processor returns decides how the output lengths are found
        # in forward(): a waveform model (wav2vec2, HuBERT, WavLM, ...) gets
        # `input_values` and the encoder's own feature extractor turns sample
        # counts into frame counts; a model whose processor already produces
        # frames (w2v-BERT) gets `input_features` with a mask over them.
        # Anything else is checked here rather than failing at the first batch.
        probe = self.processor(
            np.zeros(self.processor.sampling_rate, dtype=np.float32),
            sampling_rate=self.processor.sampling_rate,
            return_tensors="pt",
        )
        self.waveform_input = "input_values" in probe
        if self.waveform_input and not hasattr(
            self.encoder, "_get_feat_extract_output_lengths"
        ):
            raise ValueError(
                f"Frontend not supported: {type(self.encoder).__name__} takes "
                "`input_values` but has no `_get_feat_extract_output_lengths` "
                "to turn sample counts into frame counts."
            )
        if not self.waveform_input and "attention_mask" not in probe:
            raise ValueError(
                f"Frontend not supported: {type(self.processor).__name__} "
                "returns frames without an attention mask, so the number of "
                "frames per utterance is unknown."
            )

    def output_size(self) -> int:
        return self.encoder.config.hidden_size

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrapper for the transformers forward pass.

        Inputs are converted to numpy and re-encoded with the transformers processor.

        Args:
            input: Input (B, L) single channel waveform.
            input_lengths: Input lengths within batch.

        Returns:
            Tensor: Output with dimensions (B, T, D), T is the processed length,
                D is the feature dimension.
            Tensor: Output lengths within batch.
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

        feats = self.encoder(**encoded).last_hidden_state
        if self.waveform_input:
            # The processor's attention mask, when it returns one, covers
            # exactly the samples given above, so its sum would only repeat
            # `input_lengths`; the encoder's feature extractor knows how many
            # frames those samples become.
            encoded_lengths = self.encoder._get_feat_extract_output_lengths(
                input_lengths.to(device)
            )
        else:
            encoded_lengths = torch.sum(encoded.attention_mask, dim=-1)
        if torch.max(encoded_lengths) != feats.size(1):
            # truncate the sequence to the actual length
            # there is a weird bug in conformer encoder
            feats = feats[:, : torch.max(encoded_lengths), :]

        return feats, encoded_lengths

    def reload_pretrained_parameters(self):
        self.encoder.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Transformers model parameters reloaded!")
