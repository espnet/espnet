# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Voice conversion abstrast class."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class AbsVC(torch.nn.Module, ABC):
    """VC abstract class."""

    @abstractmethod
    def forward(
        self,
        # text: torch.Tensor,
        # text_lengths: torch.Tensor,
        source_feats: torch.Tensor,
        target_feats: torch.Tensor,
        source_feats_lengths: torch.Tensor,
        target_feats_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate outputs and return the loss tensor."""
        raise NotImplementedError

    @abstractmethod
    def inference(
        self,
        # text: torch.Tensor,
        source_feats: torch.Tensor,
        # target_feats: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return predicted output as a dict."""
        raise NotImplementedError

    @property
    def require_raw_speech(self):
        """Return whether or not raw_speech is required."""
        return False

    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return True
