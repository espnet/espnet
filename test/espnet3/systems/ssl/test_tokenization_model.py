"""Tests for the BEATs tokenization model used by the SSL infer stage."""

import logging

import numpy as np
import pytest
import torch

from espnet2.beats.tokenizer import BeatsTokenizer
from espnet3.systems.ssl.tokenization_model import (
    BeatsTokenizationModel,
    build_target_output,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# BeatsTokenizationModel
# | Test Name                                         | Description              |
# |---------------------------------------------------|--------------------------|
# | test_random_tokenizer_is_seeded                   | Same seed -> same codes; |
# |                                                   | codes lie in codebook.   |
# | test_batched_call_matches_single_items            | Padding does not change  |
# |                                                   | per-item codes.          |
# | test_fbank_input_uses_patch_count                 | fbank input yields one   |
# |                                                   | code per 16x16 patch.    |
# | test_int16_waveform_is_rescaled                   | Out-of-range waveform is |
# |                                                   | rescaled with a warning. |
# | test_rejects_wrong_input_rank                     | Shape mismatch raises.   |
# | test_trained_tokenizer_checkpoint                 | Loads an exported        |
# |                                                   | tokenizer checkpoint.    |
#
# build_target_output
# | Test Name                                         | Description              |
# |---------------------------------------------------|--------------------------|
# | test_build_target_output_single_and_batched       | Index-keyed records.     |

TINY_CONFIG = {
    "encoder_layers": 1,
    "encoder_embed_dim": 32,
    "encoder_ffn_embed_dim": 64,
    "encoder_attention_heads": 2,
    "embed_dim": 32,
    "conv_pos": 16,
    "conv_pos_groups": 2,
    "quant_n": 16,
    "quant_dim": 8,
}


# Small random-projection tokenizer to keep CPU tests fast.
RANDOM_CONFIG = {"seed": 45, "embed_dim": 32, "quant_n": 16, "quant_dim": 8}


def _waveform(num_samples, seed=0):
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal(num_samples)).astype(np.float32)


def test_random_tokenizer_is_seeded():
    speech = _waveform(8000)
    first = BeatsTokenizationModel(tokenizer_config=RANDOM_CONFIG, waveform_input=True)
    second = BeatsTokenizationModel(tokenizer_config=RANDOM_CONFIG, waveform_input=True)

    codes = first(speech)

    np.testing.assert_array_equal(codes, second(speech))
    assert codes.dtype == np.int64
    assert codes.min() >= 0 and codes.max() < first.codebook_size == 16


def test_batched_call_matches_single_items():
    model = BeatsTokenizationModel(tokenizer_config=RANDOM_CONFIG, waveform_input=True)
    items = [_waveform(8000, seed=1), _waveform(4000, seed=2)]

    batched = model(items)

    assert len(batched) == 2
    for item, codes in zip(items, batched):
        np.testing.assert_array_equal(codes, model(item))
    assert len(batched[1]) < len(batched[0])


def test_fbank_input_uses_patch_count():
    model = BeatsTokenizationModel(tokenizer_config=RANDOM_CONFIG, waveform_input=False)
    fbank = np.random.default_rng(0).standard_normal((160, 128)).astype(np.float32)

    codes = model(fbank)

    # 160 frames / 16 * 128 mel bins / 16 patches.
    assert codes.shape == (10 * 8,)


def test_int16_waveform_is_rescaled(caplog):
    model = BeatsTokenizationModel(tokenizer_config=RANDOM_CONFIG, waveform_input=True)
    speech = _waveform(8000)

    with caplog.at_level(logging.WARNING):
        codes = model(speech * 2**15)

    np.testing.assert_array_equal(codes, model(speech))
    assert "rescaling" in caplog.text


@pytest.mark.parametrize(
    "waveform_input, speech",
    [
        (True, np.zeros((16000, 1), dtype=np.float32)),
        (False, np.zeros(16000, dtype=np.float32)),
    ],
)
def test_rejects_wrong_input_rank(waveform_input, speech):
    model = BeatsTokenizationModel(
        tokenizer_config=RANDOM_CONFIG, waveform_input=waveform_input
    )

    with pytest.raises(ValueError, match="expects"):
        model(speech)


def test_trained_tokenizer_checkpoint(tmp_path):
    torch.manual_seed(0)
    tokenizer = BeatsTokenizer(tokenizer_config=TINY_CONFIG)
    checkpoint_path = tmp_path / "beats_tokenizer_iter1.pt"
    torch.save({"model": tokenizer.state_dict(), "cfg": TINY_CONFIG}, checkpoint_path)

    model = BeatsTokenizationModel(
        tokenizer_ckpt_path=str(checkpoint_path), waveform_input=True
    )
    codes = model(_waveform(8000))

    assert model.codebook_size == 16
    assert codes.ndim == 1 and len(codes) > 0
    assert codes.min() >= 0 and codes.max() < 16


def test_build_target_output_single_and_batched():
    assert build_target_output({}, np.array([3, 1, 2]), 7) == {
        "idx": 7,
        "target": "3 1 2",
    }
    assert build_target_output([{}, {}], [np.array([1]), np.array([4, 5])], [0, 1]) == [
        {"idx": 0, "target": "1"},
        {"idx": 1, "target": "4 5"},
    ]
