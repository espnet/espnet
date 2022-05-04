from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict
from typing import Optional
from typing import Tuple

import torch


class AbsSeparator(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:

        raise NotImplementedError

    @property
    @abstractmethod
    def num_spk(self):
        raise NotImplementedError
