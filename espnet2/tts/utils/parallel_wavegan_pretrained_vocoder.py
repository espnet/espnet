# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Wrapper class for the vocoder model trained with parallel_wavegan repo."""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch
import yaml


class ParallelWaveGANPretrainedVocoder(torch.nn.Module):
    """Wrapper class to load the vocoder trained with parallel_wavegan repo."""

    def __init__(
        self,
        model_file: Union[Path, str],
        config_file: Optional[Union[Path, str]] = None,
    ):
        """Initialize ParallelWaveGANPretrainedVocoder module."""
        super().__init__()
        try:
            from parallel_wavegan.utils import load_model
        except ImportError:
            logging.error(
                "`parallel_wavegan` is not installed. "
                "Please install via `pip install -U parallel_wavegan`."
            )
            raise
        if config_file is None:
            dirname = os.path.dirname(str(model_file))
            config_file = os.path.join(dirname, "config.yml")
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        self.fs = config["sampling_rate"]
        self.vocoder = load_model(model_file, config)
        if hasattr(self.vocoder, "remove_weight_norm"):
            self.vocoder.remove_weight_norm()
        self.normalize_before = False
        if hasattr(self.vocoder, "mean"):
            self.normalize_before = True

    @torch.no_grad()
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Generate waveform with pretrained vocoder.

        Args:
            feats (Tensor): Feature tensor (T_feats, #mels).

        Returns:
            Tensor: Generated waveform tensor (T_wav).

        """
        return self.vocoder.inference(
            feats, normalize_before=self.normalize_before,
        ).view(-1)
