#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from espnet.nets.tts_interface import TTSInterface


class DurationPredictorLoss(torch.nn.Module):
    """Duration predictor loss module

    Reference:
        - FastSpeech: Fast, Robust and Controllable Text to Speech
          (https://arxiv.org/pdf/1905.09263.pdf)
    """
    def __init__(self):
        super(DurationPredictorLoss, self).__init__()

    def forward(self):
        pass


class FeedForwardTransformer(TTSInterface, torch.nn.Module):
    """Feed Forward Transformer for TTS

    Reference:
        - FastSpeech: Fast, Robust and Controllable Text to Speech
          (https://arxiv.org/pdf/1905.09263.pdf)
    """

    def __init__(self):
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

    def forward(self):
        pass


class LengthRegularizer(torch.nn.Module):
    """Lenght regularizer module

    Reference:
        - FastSpeech: Fast, Robust and Controllable Text to Speech
          (https://arxiv.org/pdf/1905.09263.pdf)
    """
    def __init__(self):
        super(LengthRegularizer, self).__init__()

    def forward(self):
        pass


class DurationPredictor(torch.nn.Module):
    """Duration predictor module

    Reference:
        - FastSpeech: Fast, Robust and Controllable Text to Speech
          (https://arxiv.org/pdf/1905.09263.pdf)
    """
    def __init__(self):
        super(DurationPredictor, self).__init__()

    def forward(self):
        pass


