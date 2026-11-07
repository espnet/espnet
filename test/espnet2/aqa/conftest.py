from test.espnet2.aqa.test_metric_tokenizer import token_info

import pytest

from espnet2.aqa.base import UniversaBase


def _make_model(**kwargs):
    options = dict(
        input_size=8,
        metric2id={"mos": 0, "wer": 1},
        use_ref_audio=False,
        use_ref_text=False,
        embedding_size=8,
        audio_encoder_params=dict(
            num_blocks=1,
            attention_heads=2,
            linear_units=16,
            input_layer="linear",
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            attention_dropout_rate=0.0,
        ),
        cross_attention_params=dict(n_head=2, dropout_rate=0.0),
        use_mse=True,
    )
    options.update(kwargs)
    return UniversaBase(**options)


@pytest.fixture
def make_model():
    return _make_model


@pytest.fixture
def full_token_info():
    """Append whole-clip twins of both metrics after the published vocabulary."""
    info = token_info()
    for name, values in list(info["tokenizer"].items()):
        full = name + "_full"
        info["tokenizer"][full] = values
        info["offset"][full] = [len(info["VOCAB"]), len(values) + 2]
        info["VOCAB"] += [f"{full}@meta_label"]
        info["VOCAB"] += [f"{full}@{i}" for i in range(len(values) + 1)]
    return info
