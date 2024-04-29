from abc import ABC, abstractmethod
from typing import Tuple, Dict, List

import torch


class AbsPredictor(torch.nn.Module, ABC):
    """
    The abstract Predictor class for SpeechLM, which transform CoreLM output
    into logits. It shall also organzie the target so that the output logits
    is well-aligned with the target.

    It should mainly take care of two functions:
      forward: transform the input (typically CoreLM output) into logits.
      organize_target: organize target so that the output logtits is well-aligned
        with the target.

    Some usage examples:
    (1) When building a standard langauge model, this module is simply a linear
        classifier.
    (2) This module should also handle codec codes interleave, such as some 
        patterns described in MusicGen: https://arxiv.org/abs/2306.05284. In
        that case, it shall contain more than one linear classifier.
    (3) This module should also consider the dependency of codec codes. E.g.,
        for each frame, codes from deeper layers are conditioned on shallower
        layers. An example in UniAudio: UniAudio: https://arxiv.org/abs/2310.00704
    """
    def __init__(
        self,
        vocab_size: List = [],
        input_dim: int = 1,
        nq: int = 1,
        **kwargs,
    ):
        super(AbsPredictor, self).__init__()

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor = None,
        target: torch.Tensor = None,
        target_lengths: torch.Tensor = None,
        cache: dict = None,
        others: dict = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        raise NotImplementedError

    def organize_target(
        self, 
        target: torch.Tensor, 
        target_lengths: torch.Tensor,
        others: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ No change by default """
        return target, target_lengths, others

    @abstractmethod
    def get_lookup_table(self):
        """ For embedding parameter sharing """
        raise NotImplementedError
    
    ## Inference API
    def init_cache(self):
        pass
    
    def remove_cache(self):
        pass