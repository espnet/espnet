from typing import Tuple

import torch

from espnet.nets.scorer_interface import ScorerInterface


# TODO(kamo): To force the inteface we need some system like pytypes.


class LMInterface(ScorerInterface):
    def forward(self, input: torch.Tensor, hidden: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
