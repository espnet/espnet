from typing import Tuple

import pytest
import torch

from espnet2.cls.decoder.abs_decoder import AbsDecoder


def test_abs_decoder_instantiation():
    # Ensure AbsDecoder cannot be instantiated directly
    with pytest.raises(TypeError):
        AbsDecoder()


class TestDecoder(AbsDecoder):
    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = hs_pad.sum(dim=1, keepdim=True)
        return output, hlens


def test_concrete_decoder():
    decoder = TestDecoder()
    hs_pad = torch.rand(4, 10, 20)
    hlens = torch.tensor([10, 8, 6, 4])
    output, lengths = decoder(hs_pad, hlens)
    assert output.shape == (4, 1, 20)
    assert torch.equal(lengths, hlens)
