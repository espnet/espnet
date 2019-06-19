import argparse
from typing import Tuple
from typing import Union

import configargparse
import numpy as np
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.asr_interface import FrontendASRInterface
from espnet.nets.pytorch_backend.frontends.feature_transform import FeatureTransform
from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.utils.cli_utils import strtobool


class FrontendASR(FrontendASRInterface, torch.nn.Module):
    """An implementation of FrontendASR with WPE and MVDR"""
    @staticmethod
    def add_arguments(parser: Union[argparse.ArgumentParser,
                                    configargparse.ArgumentParser]):
        # WPE related
        parser.add_argument('--use-wpe', type=strtobool, default=False,
                            help='Apply Weighted Prediction Error')
        parser.add_argument('--wtype', default='blstmp', type=str,
                            choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                     'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                            help='Type of encoder network architecture '
                                 'of the mask estimator for WPE. '
                                 '')
        parser.add_argument('--wlayers', type=int, default=2,
                            help='')
        parser.add_argument('--wunits', type=int, default=300,
                            help='')
        parser.add_argument('--wprojs', type=int, default=300,
                            help='')
        parser.add_argument('--wdropout-rate', type=float, default=0.0,
                            help='')
        parser.add_argument('--wpe-taps', type=int, default=5,
                            help='')
        parser.add_argument('--wpe-delay', type=int, default=3,
                            help='')
        parser.add_argument('--use-dnn-mask-for-wpe', type=strtobool,
                            default=False,
                            help='Use DNN to estimate the power spectrogram. '
                                 'This option is experimental.')
        # Beamformer related
        parser.add_argument('--use-beamformer', type=strtobool,
                            default=True, help='')
        parser.add_argument('--btype', default='blstmp', type=str,
                            choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                     'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                            help='Type of encoder network architecture '
                                 'of the mask estimator for Beamformer.')
        parser.add_argument('--blayers', type=int, default=2,
                            help='')
        parser.add_argument('--bunits', type=int, default=300,
                            help='')
        parser.add_argument('--bprojs', type=int, default=300,
                            help='')
        parser.add_argument('--badim', type=int, default=320,
                            help='')
        parser.add_argument('--ref-channel', type=int, default=-1,
                            help='The reference channel used for beamformer. '
                                 'By default, the channel is estimated by DNN.')
        parser.add_argument('--bdropout-rate', type=float, default=0.0,
                            help='')
        # Feature transform: Normalization
        parser.add_argument('--stats-file', type=str, default=None,
                            help='The stats file for the feature normalization')
        parser.add_argument('--apply-uttmvn', type=strtobool, default=True,
                            help='Apply utterance level mean '
                                 'variance normalization.')
        parser.add_argument('--uttmvn-norm-means', type=strtobool,
                            default=True, help='')
        parser.add_argument('--uttmvn-norm-vars', type=strtobool, default=False,
                            help='')
        # Feature transform: Fbank
        parser.add_argument('--fbank-fs', type=int, default=16000,
                            help='The sample frequency used for '
                                 'the mel-fbank creation.')
        parser.add_argument('--n-mels', type=int, default=80,
                            help='The number of mel-frequency bins.')
        parser.add_argument('--fbank-fmin', type=float, default=0.,
                            help='')
        parser.add_argument('--fbank-fmax', type=float, default=None,
                            help='')

    def __init__(self, idim: int, args: argparse.Namespace,
                 asr_model: torch.nn.Module):
        torch.nn.Module.__init__(self)
        assert isinstance(asr_model, ASRInterface), type(asr_model)
        self.asr_model = asr_model

        self.frontend = Frontend(
            idim=idim,
            # WPE options
            use_wpe=args.use_wpe,
            wtype=args.wtype,
            wlayers=args.wlayers,
            wunits=args.wunits,
            wprojs=args.wprojs,
            wdropout_rate=args.wdropout_rate,
            taps=args.wpe_taps,
            delay=args.wpe_delay,
            use_dnn_mask_for_wpe=args.use_dnn_mask_for_wpe,

            # Beamformer options
            use_beamformer=args.use_beamformer,
            btype=args.btype,
            blayers=args.blayers,
            bunits=args.bunits,
            bprojs=args.bprojs,
            badim=args.badim,
            ref_channel=args.ref_channel,
            bdropout_rate=args.bdropout_rate)

        n_fft = (idim - 1) * 2
        self.feature_transform = FeatureTransform(
            # Mel options,
            fs=args.fbank_fs,
            n_fft=n_fft,
            n_mels=args.n_mels,
            fmin=args.fbank_fmin,
            fmax=args.fbank_fmax,

            # Normalization
            stats_file=args.stats_file,
            apply_uttmvn=args.apply_uttmvn,
            uttmvn_norm_means=args.uttmvn_norm_means,
            uttmvn_norm_vars=args.uttmvn_norm_vars)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        xs_pad = to_torch_tensor(xs_pad)
        enhanced, hlens, mask = self.frontend(xs_pad, ilens)
        hs_pad, hlens = self.feature_transform(enhanced, hlens)
        return self.asr_model(hs_pad, hlens, ys_pad)

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        prev = self.training
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[::self.asr_model.subsample[0], :]
        h = to_device(self, to_torch_tensor(x).float())
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # The concrete class for Fronend is here: espnet.nets.pytorch_backend.frontend_asr.py
        enhanced, hlens, mask = self.frontend(hs, ilens)
        hs, hlens = self.feature_transform(enhanced, hlens)

        if prev:
            self.train()
        return self.asr_model.recognize(hs[0], recog_args, char_list, rnnlm=None)

    def recognize_batch(self, xs, recog_args, char_list, rnnlm=None):
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.asr_model.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        enhanced, hlens, mask = self.frontend(xs_pad, ilens)
        hs_pad, hlens = self.feature_transform(enhanced, hlens)
        if prev:
            self.train()
        return self.asr_model.recognize_batch(hs_pad, recog_args, char_list, rnnlm=None)

    def enhance(self, xs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forwarding only the frontend stage

        :param ndarray xs: input acoustic feature (T, C, F)
        """
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.asr_model.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)
        enhanced, hlens, mask = self.frontend(xs_pad, ilens)
        if prev:
            self.train()
        if mask is not None:
            mask = mask.cpu().numpy()
        return enhanced.cpu().numpy(), hlens.cpu().numpy(), mask

    def calculate_all_attentions(self, xs: list, ilens: np.ndarray, ys: list):
        xs = to_torch_tensor(xs)
        enhanced, hlens, mask = self.frontend(xs, ilens)
        hs, hlens = self.feature_transform(enhanced, hlens)
        return self.asr_model.calculate_all_attentions(hs, hlens, ys)
