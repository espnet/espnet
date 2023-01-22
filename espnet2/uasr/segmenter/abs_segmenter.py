"""
Segmenter definition for UASR task

Practially, the output of the generator (in frame-level) may
predict the same phoneme for consecutive frames, which makes
it too easy for the discriminator. So, the segmenter here is
to merge frames with a similar prediction from the generator output.
"""

from abc import ABC, abstractmethod

import torch


class AbsSegmenter(torch.nn.Module, ABC):
    @abstractmethod
    def pre_segment(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def logit_segment(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
