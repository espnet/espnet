import pytest

from espnet2.universa.base import UniversaBase


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
