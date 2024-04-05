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
            print(
                "The version provided by espnet is not up-to-dated, not covering these"
            )
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e

        if config_file is None:
            dirname = os.path.dirname(str(model_file))
            config_file = os.path.join(dirname, "config.yml")

        with open(config_file) as f:
            config = yaml.safe_load(f)
        self.fs = config["sampling_rate"]
        self.vocoder = CodeHiFiGANVocoder(model_file, config)

    @torch.no_grad()
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Generate waveform with pretrained vocoder.
        Args:
            feats (Tensor): Feature tensor (T_feats, #mels).

        Returns:
            Tensor: Generated waveform tensor (T_wav).
        """
        input_discrete_unit_reshaped = feats.view(1, -1)
        x = {
            "code": input_discrete_unit_reshaped,
        }
        if torch.cuda.is_available():
            x = {k: v.cuda() for k, v in x.items()}

        return self.vocoder.forward(x).detach()
