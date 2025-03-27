<<<<<<< HEAD
from abc import ABC, abstractmethod
from typing import Tuple
=======
#!/usr/bin/env python3

# Copyright 2025 William Chen
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
>>>>>>> e72ffd9f44396fd97121dc5d38893d66b18756cc

import torch


<<<<<<< HEAD
class AbsLoss(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        xs_pad: torch.Tensor,
        ys_pad: torch.Tensor,
    ) -> torch.Tensor:
=======
class AbsSSLLoss(torch.nn.Module, ABC):
    """Abstract loss class for encoder-only SSL model.

    Each loss class must contain 2 attributes:

        - self.util_attributes (List): functions that need to be
            performed on encoder input (masking, noise, etc).
        - self.required_inputs(List): data names needed to perform
            loss calculation.
    """

    @abstractmethod
    def forward(
        self,
        encoder_output: List,
        encoder_output_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward for an SSL objective

        Args:
            encoder_output (List): List of encoded sequences (B, T, D) from each layer
            encoder_output_lengths (Tensor): Lengths of batched encoder sequences (B,),

        """
>>>>>>> e72ffd9f44396fd97121dc5d38893d66b18756cc
        raise NotImplementedError
