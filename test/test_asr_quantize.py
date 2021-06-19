# Copyright 2021 Gaopeng Xu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import pytest
import torch

from espnet.nets.asr_interface import dynamic_import_asr


@pytest.mark.parametrize(
    "name, backend",
    [(nn, backend) for nn in ("transformer", "rnn") for backend in ("pytorch",)],
)
def test_asr_quantize(name, backend):
    model = dynamic_import_asr(name, backend).build(
        10, 10, mtlalpha=0.123, adim=4, eunits=2, dunits=2, elayers=1, dlayers=1
    )
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    assert quantized_model.state_dict()
