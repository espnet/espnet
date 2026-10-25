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
def legacy_token_info():
    """Tiny ANCHOR-style vocabulary: legacy categories and one shifted metric."""
    tokenizers = {
        "mos": [0.0, 1.0, 2.0, 3.0],
        "language": ["eng", "jpn", "zho"],
        "bg_event": ["none", "music", "vehicle"],
    }
    vocab, offset = [], {}
    for key, values in tokenizers.items():
        offset[key] = [len(vocab), len(values) + 2]
        vocab += [key + "@meta_label"] + [f"{key}@{i}" for i in range(len(values) + 1)]
    return dict(
        tokenizer=tokenizers,
        VOCAB=vocab,
        offset=offset,
        categorical_scheme="legacy",
        shifted_categorical_metrics=["bg_event"],
    )
