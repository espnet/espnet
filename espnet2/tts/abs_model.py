from abc import abstractmethod
from typing import Tuple

import torch

from espnet2.train.abs_espnet_model import AbsESPNetModel


class AbsTTSModel(AbsESPNetModel):
    @abstractmethod
    def forward(self,
                input: torch.Tensor,
                input_lengths: torch.Tensor,
                output: torch.Tensor,
                output_lengths: torch.Tensor,
                spembs: torch.Tensor = None,
                spembs_lengths: torch.Tensor = None,
                spcs: torch.Tensor = None,
                spcs_lengths: torch.Tensor = None) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def inference(self,
                  input: torch.Tensor,
                  threshold: float,
                  minlenratio: float,
                  maxlenratio: float,
                  spembs: torch.Tensor = None,
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
