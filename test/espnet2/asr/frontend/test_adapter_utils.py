import argparse

import numpy
import pytest
import torch
from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer

from espnet2.asr.frontend.adapter_utils import *


def test_add_adapters_wav2vec2():
    model = {}
    model.update(
        {
            "model": {
                "encoder": {
                    "layers": [
                        TransformerSentenceEncoderLayer(
                            10, 20, 2, 0.1, 0.1, 0.1, "relu", False
                        ),
                        TransformerSentenceEncoderLayer(
                            10, 20, 2, 0.1, 0.1, 0.1, "relu", False
                        ),
                        TransformerSentenceEncoderLayer(
                            10, 20, 2, 0.1, 0.1, 0.1, "relu", False
                        ),
                    ]
                }
            }
        }
    )
    x = torch.randn(1, 10)
    original = model(x)
    model = add_adapters_wav2vec2(model, 122, [0, 2])
    assert isinstance(
        model.model.encoder.layers[0], AdapterTransformerSentenceEncoderLayer
    )
    assert isinstance(model.model.encoder.layers[1], TransformerSentenceEncoderLayer)
    assert isinstance(
        model.model.encoder.layers[2], AdapterTransformerSentenceEncoderLayer
    )

    curr = model(x)
    assert curr != original
