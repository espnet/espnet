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
    """
        Wrapper class to load the vocoder trained with parallel_wavegan repo.

    This class is designed to facilitate the loading and utilization of a vocoder
    model that has been trained using the Parallel WaveGAN framework. It
    integrates with the PyTorch framework and allows for generating waveforms
    from input features.

    Attributes:
        fs (int): Sampling rate of the vocoder.
        vocoder (torch.nn.Module): The loaded vocoder model.

    Args:
        model_file (Union[Path, str]): Path to the model file.
        config_file (Optional[Union[Path, str]]): Path to the configuration file.
            If None, the configuration is loaded from the same directory as the
            model file, with the name "config.yml".

    Raises:
        ImportError: If the `parallel_wavegan` package is not installed.
        FileNotFoundError: If the configuration file does not exist.

    Examples:
        >>> vocoder = ParallelWaveGANPretrainedVocoder("path/to/model/file")
        >>> waveform = vocoder(torch.randn(100, 80))  # Assuming 80 mel features.

    Note:
        Ensure that the `parallel_wavegan` library is installed for this class
        to function correctly.
    """

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
        """
            Generate waveform with pretrained vocoder.

        This method takes a feature tensor as input and generates a corresponding
        waveform tensor using the pretrained vocoder model.

        Args:
            feats (torch.Tensor): Feature tensor of shape (T_feats, #mels), where
                T_feats is the number of frames and #mels is the number of mel
                frequency bins.

        Returns:
            torch.Tensor: Generated waveform tensor of shape (T_wav), where T_wav
                is the length of the output waveform.

        Examples:
            >>> import torch
            >>> vocoder = ParallelWaveGANPretrainedVocoder("path/to/model/file")
            >>> features = torch.randn(100, 80)  # Example mel features
            >>> waveform = vocoder(features)
            >>> print(waveform.shape)  # Output shape should be (T_wav,)
        """
        return self.vocoder.inference(
            feats,
            normalize_before=self.normalize_before,
        ).view(-1)
