import pytest
from distutils.version import LooseVersion
import os

import torch

from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.default import DefaultFrontend


list_frontends = [DefaultFrontend(fs="16k"), DefaultFrontend(fs="16k")]


def test_frontend_init():
    frontend = FusedFrontend(fs="16k", align_method="linear_projection", proj_dim=100, frontends=list_frontends)
    assert len(frontend.frontends) == 2
    assert len(frontend.factors) == len(frontend.frontends)
    assert frontend[0].frontend_type == "default"


def test_frontend_output_size():
    frontend = FusedFrontend(fs="16k", align_method="linear_projection", proj_dim=100, frontends=list_frontends)
    assert frontend.output_size() == 100 * len(list_frontends)


def test_frontend_backward():
    frontend = FusedFrontend(fs="16k", align_method="linear_projection", proj_dim=100, frontends=list_frontends)
    x = torch.randn(2, 300, requires_grad=True)
    x_lengths = torch.LongTensor([300, 89])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()


