import os
from pathlib import Path
from typing import Optional, Union

import torch
import yaml


class FairseqHifiGANPretrainedVocoder(torch.nn.Module):
    """FairSeq HifiGAN pretrain encoder module, only used for pretraining stage"""

    def __init__(
        self,
        model_file: Union[Path, str],
        config_file: Optional[Union[Path, str]] = None,
    ):
        
        """Initialize ParallelWaveGANPretrainedVocoder module."""
        super().__init__()
        try:
            from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e
        
        if config_file is None:
            dirname = os.path.dirname(str(model_file))
            config_file = os.path.join(dirname, "config.yml")

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml)
        self.fs = config["sampling_rate"]
        self.vocoder = CodeHiFiGANVocoder(model_file, config)
        

    @torch.no_grad()
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Generate waveform with pretrained vocoder."""
        return self.vocoder.forward(
            feats,
        )