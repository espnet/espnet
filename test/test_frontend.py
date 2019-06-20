import argparse
from typing import Union

import numpy as np
import pytest
import torch
from torch_complex.tensor import ComplexTensor

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.frontend_asr import FrontendASR
from espnet.nets.pytorch_backend.frontends.dnn_beamformer import DNN_Beamformer
from espnet.nets.pytorch_backend.frontends.dnn_wpe import DNN_WPE
from espnet.nets.pytorch_backend.frontends.feature_transform import FeatureTransform
from espnet.nets.pytorch_backend.frontends.feature_transform import GlobalMVN
from espnet.nets.pytorch_backend.frontends.feature_transform import LogMel
from espnet.nets.pytorch_backend.frontends.feature_transform import UtteranceMVN
from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet.nets.pytorch_backend.nets_utils import pad_list


def prepare_inputs(nfft=512, mode='pytorch', ilens=[150, 100], olens=[4, 3], is_cuda=False):
    np.random.seed(1)
    assert len(ilens) == len(olens)
    xs = [np.random.randn(ilen, 3, nfft // 2 + 1).astype(np.float32) for ilen in ilens]
    ys = [np.random.randint(1, 5, olen).astype(np.int32) for olen in olens]
    ilens = np.array([x.shape[0] for x in xs], dtype=np.int32)

    if mode == "chainer":
        raise NotImplementedError

    elif mode == "pytorch":
        ilens = torch.from_numpy(ilens).long()
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], -1)
        if is_cuda:
            xs_pad = xs_pad.cuda()
            ilens = ilens.cuda()
            ys_pad = ys_pad.cuda()

        return ComplexTensor(xs_pad, xs_pad), ilens, ComplexTensor(ys_pad, ys_pad)
    else:
        raise ValueError("Invalid mode")


class MockAsr(ASRInterface, torch.nn.Module):
    def __init__(self, xs_shape, ys_shape, ilens):
        torch.nn.Module.__init__(self)
        self.xs_shape, self.ys_shape, self.ilens = \
            xs_shape, ys_shape, ilens
        self.subsample = [1]

    def forward(self, xs: torch.Tensor, ilens: torch.Tensor, ys: torch.Tensor):
        assert list(xs.shape) == self.xs_shape
        assert list(ys.shape) == self.ys_shape
        assert list(ilens) == list(self.ilens)

    def recognize(self,
                  x: Union[torch.Tensor, np.ndarray],
                  recog_args: argparse.Namespace,
                  char_list: list = None,
                  rnnlm: torch.nn.Module = None):
        assert list(x.shape) == self.xs_shape

    def recognize_batch(self,
                        x: Union[torch.Tensor, np.ndarray],
                        recog_args: argparse.Namespace,
                        char_list: list = None,
                        rnnlm: torch.nn.Module = None):
        assert list(x.shape) == self.xs_shape

    def calculate_all_attentions(self, xs: list, ilens: np.ndarray, ys: list):
        assert list(xs.shape) == self.xs_shape
        assert list(ys.shape) == self.ys_shape
        assert list(ilens) == list(self.ilens)

    @property
    def attention_plot_class(self):
        from espnet.asr.asr_utils import PlotAttentionReport
        return PlotAttentionReport


def test_FrontendASR_forward():
    parser = argparse.ArgumentParser()
    # Test1: add_arguments and parse
    FrontendASR.add_arguments(parser)
    nmels = 80
    args, _ = parser.parse_known_args(['--use-wpe', 'true',
                                       '--use-beamformer', 'true',
                                       '--n-mels', str(nmels)])

    nfft = 512
    xs, ilens, ys = prepare_inputs(nfft)

    # Test2: Instantiate
    asr_model = MockAsr([2, 150, nmels], [2, 4], ilens)
    model = FrontendASR(nfft // 2 + 1, args, asr_model)

    # Test3: Forward
    model.forward(xs, ilens, ys)


def test_FrontendASR_recognize():
    defaults = dict(
        nbest=5,
        beam_size=3,
        penalty=0.5,
        maxlenratio=1.0,
        minlenratio=0.0,
        ctc_weight=0.2,
        streaming_min_blank_dur=10,
        streaming_onset_margin=2,
        streaming_offset_margin=2,
        verbose=2,
        char_list=[u"あ", u"い", u"う", u"え", u"お"],
        ctc_type="warpctc",
        sym_space="<space>",
        sym_blank="<blank>")
    parser = argparse.ArgumentParser()
    # Test1: add_arguments and parse
    FrontendASR.add_arguments(parser)
    nmels = 80
    args, _ = parser.parse_known_args(['--use-wpe', 'true',
                                       '--use-beamformer', 'true',
                                       '--n-mels', str(nmels)])
    for k, v in defaults.items():
        setattr(args, k, v)
    nfft = 512
    # Test2: Instantiate
    asr_model = MockAsr([3, nmels], None, None)
    model = FrontendASR(nfft // 2 + 1, args, asr_model)
    x = np.random.randn(3, nfft // 2 + 1) + 1j * np.random.randn(3, nfft // 2 + 1)
    model.recognize(x, args, args.char_list)  # decodable

    if hasattr(model, 'recognize_batch'):
        asr_model = MockAsr([2, 3, nmels], None, None)
        model = FrontendASR(nfft // 2 + 1, args, asr_model)
        x = np.random.randn(2, 3, nfft // 2 + 1) + 1j * np.random.randn(2, 3, nfft // 2 + 1)
        model.recognize_batch(x, args, args.char_list)  # decodable


def test_FrontendASR_calculate_all_attentions():
    parser = argparse.ArgumentParser()
    # Test1: add_arguments and parse
    FrontendASR.add_arguments(parser)
    nmels = 80
    args, _ = parser.parse_known_args(['--use-wpe', 'true',
                                       '--use-beamformer', 'true',
                                       '--n-mels', str(nmels)])

    nfft = 512
    xs, ilens, ys = prepare_inputs(nfft)

    # Test2: Instantiate
    asr_model = MockAsr([2, 150, nmels], [2, 4], ilens)
    model = FrontendASR(nfft // 2 + 1, args, asr_model)

    # Test3: Forward
    model.calculate_all_attentions(xs, ilens, ys)


def test_DNN_Beamformer_forward_backward():
    model = DNN_Beamformer(
        bidim=257,
        btype='blstmp',
        blayers=3,
        bunits=300,
        bprojs=320,
        dropout_rate=0.0,
        badim=320,
        ref_channel=-1,
        beamformer_type='mvdr')

    # (Batch, Channel, TimeFrame, FreqBin)
    xr = torch.randn(2, 10, 3, 257, dtype=torch.float)
    xi = torch.randn(2, 10, 3, 257, dtype=torch.float)
    x = ComplexTensor(xr, xi)
    ilens = torch.LongTensor([10, 10])
    y, _, _ = model(x, ilens)
    return y.abs().sum().backward()


def test_DNN_WPE_forward_backward():
    model = DNN_WPE(
        wtype='blstmp',
        widim=257,
        wlayers=3,
        wunits=300,
        wprojs=320,
        dropout_rate=0.0,
        taps=5,
        delay=3,
        use_dnn_mask=True,
        iterations=1,
        normalization=False)

    # (Batch, Channel, TimeFrame, FreqBin)
    xr = torch.randn(2, 10, 3, 257, dtype=torch.float)
    xi = torch.randn(2, 10, 3, 257, dtype=torch.float)
    x = ComplexTensor(xr, xi)
    ilens = torch.LongTensor([10, 10])
    y, _, _ = model(x, ilens)
    return y.abs().sum().backward()


def test_Frontend_forward_backward():
    model = Frontend(
        idim=257,
        # WPE options
        use_wpe=True,
        wtype='blstmp',
        wlayers=3,
        wunits=300,
        wprojs=320,
        wdropout_rate=0.0,
        taps=5,
        delay=3,
        use_dnn_mask_for_wpe=True,

        # Beamformer options
        use_beamformer=True,
        btype='blstmp',
        blayers=3,
        bunits=300,
        bprojs=320,
        badim=320,
        ref_channel=-1,
        bdropout_rate=0.0)

    # (Batch, Channel, TimeFrame, FreqBin)
    xr = torch.randn(2, 10, 3, 257, requires_grad=True, dtype=torch.float)
    xi = torch.randn(2, 10, 3, 257, requires_grad=True, dtype=torch.float)
    x = ComplexTensor(xr, xi)
    ilens = torch.LongTensor([10, 10])
    y, _, _ = model(x, ilens)
    return y.abs().sum().backward()


def test_LogMel_forward_backward():
    model = LogMel(fs=16000, n_fft=512, n_mels=80,
                   fmin=0, fmax=None, htk=True, norm=1)
    # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
    feat = torch.randn(2, 10, 257, requires_grad=True, dtype=torch.float)
    ilens = torch.LongTensor([10, 10])
    y, _ = model(feat, ilens)

    assert y.shape == (2, 10, 80), y.shape
    # (Batch, Channel, TimeFrame, FreqBin)
    return y.sum().backward()


@pytest.mark.parametrize('norm_means', [True, False])
@pytest.mark.parametrize('norm_vars', [True, False])
def test_GlobalMVN_forward_backward(tmpdir, norm_means, norm_vars):
    p = tmpdir.join("stats.npy")
    featdim = 80
    # stats_array: The first half is sum, the second half is sum square,
    # and the last is number of samples. This starange format comes from kaldi style.
    nsample = 100
    feats = np.random.rand(nsample, featdim)
    stats_array = np.concatenate([feats.sum(axis=0),
                                  (feats ** 2).sum(axis=0),
                                  np.array([nsample])])
    np.save(str(p), stats_array)
    model = GlobalMVN(
        str(p),
        norm_means=norm_means,
        norm_vars=norm_vars,
        eps=1.0e-20)

    # feat: (B, T, D1)
    feat = torch.randn(2, 10, featdim, requires_grad=True, dtype=torch.float)
    ilens = torch.LongTensor([10, 10])
    y, _ = model(feat, ilens)
    assert y.shape == (2, 10, 80), y.shape
    # (Batch, Channel, TimeFrame, FreqBin)
    return y.sum().backward()


@pytest.mark.parametrize('norm_means', [True, False])
@pytest.mark.parametrize('norm_vars', [True, False])
def test_UtteranceMVN_forward_backward(norm_means, norm_vars):
    model = UtteranceMVN(
        norm_means=norm_means,
        norm_vars=norm_vars,
        eps=1.0e-20)
    # feat: (B, T, D1)
    feat = torch.randn(2, 10, 80, requires_grad=True, dtype=torch.float)
    ilens = torch.LongTensor([10, 10])
    y, _ = model(feat, ilens)
    assert y.shape == (2, 10, 80), y.shape
    # (Batch, Channel, TimeFrame, FreqBin)
    return y.sum().backward()


@pytest.mark.parametrize('_global', [True, False])
def test_FeatureTransform_forward_backward(tmpdir, _global):
    if _global:
        p = tmpdir.join("stats.npy")
        stats_file = str(p)
        featdim = 80
        # stats_array: The first half is sum, the second half is sum square,
        # and the last is number of samples. This starange format comes from kaldi style.
        nsample = 100
        feats = np.random.rand(nsample, featdim)
        stats_array = np.concatenate([feats.sum(axis=0),
                                      (feats ** 2).sum(axis=0),
                                      np.array([nsample])])
        np.save(stats_file, stats_array)
    else:
        stats_file = None

    nfft = 512
    model = FeatureTransform(
        # Mel options,
        fs=16000,
        n_fft=nfft,
        n_mels=80,
        fmin=0.0,
        fmax=None,
        # Normalization
        stats_file=stats_file,
        apply_uttmvn=not _global,
        uttmvn_norm_means=True,
        uttmvn_norm_vars=False)

    # (Batch, Channel, TimeFrame, FreqBin)
    xr = torch.randn(2, 10, 3, nfft // 2 + 1, requires_grad=True, dtype=torch.float)
    xi = torch.randn(2, 10, 3, nfft // 2 + 1, requires_grad=True, dtype=torch.float)
    x = ComplexTensor(xr, xi)
    ilens = torch.LongTensor([10, 10])
    y, _ = model(x, ilens)
    return y.abs().sum().backward()
