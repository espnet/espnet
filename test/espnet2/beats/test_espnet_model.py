import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.encoder.beats_encoder import (
    BeatsConfig,
    BeatsEncoder,
    BeatsPretrainingPredictor,
)
from espnet2.beats.espnet_model import BeatsPretrainModel

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


def test_forward_backward():
    if not is_torch_1_12_1_plus:
        return
    beats_config = BeatsConfig(
        cfg={
            "encoder_layers": 2,
            "encoder_embed_dim": 128,
            "decoder_embed_dim": 1024,
            "encoder_attention_heads": 4,
            "codebook_vocab_size": 24,
        }
    )
    encoder = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        is_pretraining=True,
    )
    predictor = BeatsPretrainingPredictor(beats_config=beats_config)
    model = BeatsPretrainModel(encoder=encoder, decoder=predictor)
    inputs = dict(
        speech=torch.randn(2, 16000, 1, dtype=torch.float32),
        speech_lengths=torch.tensor([16000, 8000], dtype=torch.long),
        target=torch.randint(0, 24, [2, 48], dtype=torch.long),
        target_lengths=torch.tensor([48, 48], dtype=torch.long),
    )
    loss, stats, weight = model(**inputs)
    assert loss is not None
    loss.backward()
