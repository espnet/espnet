import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.encoder.beats_encoder import (
    BeatsEncoder,
    BeatsPretrainingPredictor,
)
from espnet2.speechlm.tokenizer.beats_tokenizer import (
    BeatsTokenizer,
    BeatsTokenizerPretrainingPredictor,
)
from espnet2.beats.espnet_model import BeatsPretrainModel, BeatsTokenizerPretrainModel

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


def test_forward_backward_beats_pretrain_model():
    if not is_torch_1_12_1_plus:
        return
    beats_config = {
        "encoder_layers": 2,
        "encoder_embed_dim": 128,
        "decoder_embed_dim": 1024,
        "encoder_attention_heads": 4,
        "codebook_vocab_size": 24,
    }
    encoder = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        is_pretraining=True,
    )
    predictor = BeatsPretrainingPredictor(beats_config=beats_config)
    model = BeatsPretrainModel(encoder=encoder, decoder=predictor)
    inputs = dict(
        speech=torch.randn(2, 16000, dtype=torch.float32, requires_grad=True),
        speech_lengths=torch.tensor([16000, 8000], dtype=torch.long),
        target=torch.randint(0, 24, [2, 48], dtype=torch.long),
        target_lengths=torch.tensor([48, 48], dtype=torch.long),
    )
    loss, stats, weight = model(**inputs)
    assert loss is not None
    loss.backward()


def test_forward_backward_beats_tokenizer_pretrain_model():
    if not is_torch_1_12_1_plus:
        return

    beats_config = {
        "encoder_layers": 2,
        "encoder_embed_dim": 768,
        "decoder_embed_dim": 768,
        "encoder_attention_heads": 4,
        "decoder_layers": 2,
    }
    tokenizer_config = beats_config.copy()
    tokenizer_config["codebook_vocab_size"] = 24
    encoder = BeatsTokenizer(
        tokenizer_config=tokenizer_config,
    )
    predictor = BeatsTokenizerPretrainingPredictor(tokenizer_config=tokenizer_config)
    teacher = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        is_pretraining=False,
    )
    model = BeatsTokenizerPretrainModel(
        encoder=encoder, decoder=predictor, teacher=teacher
    )
    inputs = dict(
        speech=torch.randn(2, 16000, dtype=torch.float32, requires_grad=True),
        speech_lengths=torch.tensor([16000, 8000], dtype=torch.long),
    )
    loss, stats, weight = model(**inputs)
    assert loss is not None
    loss.backward()
