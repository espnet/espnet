#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
from abc import ABC, abstractmethod

class AbsTransformer(torch.nn.Module, ABC):
    """ 
    
    Transformer body implementation.
    It should take care of (1 ) Stacked Transforemr layers; (2) Positional Encoding

    """
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs):
        """ Forward function, should be compatible with both training and inference """
        raise NotImplementedError
    
    @abstractmethod
    def init(self):
        """ Initialize cache before auto-regressive prediction """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        """ Clear cache after auto-regressive prediction """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def n_ctx(self):
        """ Maximum allowed length of context """
        raise NotImplementedError
    

