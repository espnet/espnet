#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch


class AbsCoreLM(torch.nn.Module, ABC):
    """
    The abstract CoreLM class for SpeechLM, which is the major component of SpeechLM.
    Most trainable parameters belong to this module.
    
    In philosophy, the CoreLM are stacked Transformer Layers, whose input and output
    are both dense representations (It also contains Positional Encoding). Other 
    modules, such as embeddings, language model predictors, should not be included 
    in this module.

    It shall support several styles:
    Auto-Regressive (AR): frames are processed frame-by-frame. It could be either
      encoder-decoder or decoder-only, but the decoder should always be causal.
      several representatives:
          SpearTTS: https://arxiv.org/abs/2302.03540
          MusicGen: https://arxiv.org/abs/2306.05284
          UniAudio: https://arxiv.org/abs/2310.00704

    Non-Auto-Regressive (NAR): frames are processed in parallel style. It should
      mainly be decoder-only architecture. Several representitives:
          SoundStorm: https://arxiv.org/abs/2305.09636

    Auto-Regressive + Non-Auto-Regressive (AR + NRA): Hybrid of AR and NAR. 
      several representitives:
          Vall-E: https://arxiv.org/abs/2301.02111
    
    For developers: If you build the model architecture yourself, try to keep the
      model file dependency within espnet2.speechlm so it would be easier to 
      transplant your model in the future.
    
    It shall also support external-sourced pre-trained models, especially those from
      HuggingFace.
    """

    @abstractmethod
    def forward(
        self,
        decoder_input: torch.Tensor,
        decoder_input_lengths: torch.Tensor = None,
        encoder_input: torch.Tensor = None,
        encoder_input_lengths: torch.Tensor = None,
        cache: Dict = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        raise NotImplementedError

    @property
    def model_dim(self) -> int:
        return getattr(self, "_model_dim")
    
    @property
    def encoder_decoder_format(self) -> int:
        return getattr(self, "_encoder_decoder_format")
    
    # cache is for inference
    def init_cache(self):
        pass
    
    def remove_cache(self):
        pass
