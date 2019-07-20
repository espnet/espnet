import argparse
from typing import List
from typing import Union

import chainer
import configargparse
import numpy as np
import torch


class ASRInterface(object):
    """ASR Interface for ESPnet model implementation"""

    @staticmethod
    def add_arguments(parser: Union[argparse.ArgumentParser,
                                    configargparse.ArgumentParser]):
        return parser

    def register_frontend(frotnend: torch.nn.Module):
        raise NotImplementedError("register_frontend method is not implemented")

    def forward(self, xs: Union[torch.Tensor, List[chainer.Variable]],
                ilens: Union[torch.Tensor, List[int]],
                ys: Union[torch.Tensor, List[chainer.Variable]]):
        '''compute loss for training

        :param xs:
            For pytorch, batch of padded source sequences torch.Tensor (B, Tmax, idim)
            For chainer, list of source sequences chainer.Variable
        :param ilens: batch of lengths of source sequences (B)
            For pytorch, torch.Tensor
            For chainer, list of int
        :param ys:
            For pytorch, batch of padded source sequences torch.Tensor (B, Lmax)
            For chainer, list of source sequences chainer.Variable
        :return: loss value
        :rtype: torch.Tensor for pytorch, chainer.Variable for chainer
        '''
        raise NotImplementedError("forward method is not implemented")

    def recognize(self,
                  x: Union[torch.Tensor, np.ndarray],
                  recog_args: argparse.Namespace,
                  char_list: list = None,
                  rnnlm: torch.nn.Module = None):
        '''recognize x for evaluation

        :param ndarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        raise NotImplementedError("recognize method is not implemented")

    def calculate_all_attentions(self, xs: list, ilens: np.ndarray, ys: list):
        '''attention calculation

        :param list xs_pad: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: attention weights (B, Lmax, Tmax)
        :rtype: float ndarray
        '''
        raise NotImplementedError("calculate_all_attentions method is not implemented")

    @property
    def attention_plot_class(self):
        from espnet.asr.asr_utils import PlotAttentionReport
        return PlotAttentionReport


class FrontendASRInterface:
    """Frontend part of ASR Interface

    >>> frontend = FrontendASR(infeatdim, args)
    >>> asr_model = E2E(frontend.featdim, args)
    >>> asr_model.register_asr(frontend)

    """

    def __init__(self, idim: int, args: argparse.Namespace):
        raise NotImplementedError("__init__ method is not implemented")

    @property
    def featdim(self) -> int:
        """The dimention of the output feature from Frontend block,

        Frontend -> Feature(Batch, time, featdim) -> E2E class
        """
        raise NotImplementedError("featdim method is not implemented")

    def forward(self, xs_pad, ilens):
        raise NotImplementedError('forward method is not implemented')

    def enhance(self, xs: np.ndarray):
        """Forwarding only the frontend stage

        :param ndarray xs: input acoustic feature (T, C, F)
        """
        raise NotImplementedError("enhance method is not implemented")
