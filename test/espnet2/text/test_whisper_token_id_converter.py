import pytest

from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter

pytest.importorskip("whisper")


@pytest.fixture(params=["whisper_multilingual"])
def whisper_token_id_converter(request):
    return OpenAIWhisperTokenIDConverter(request.param)


def test_init_invalid():
    with pytest.raises(ValueError):
        OpenAIWhisperTokenIDConverter("whisper_aaa")


def test_init_en():
    id_converter = OpenAIWhisperTokenIDConverter("whisper_en")
    assert id_converter.get_num_vocabulary_size() == 50363


def test_ids2tokens(whisper_token_id_converter: OpenAIWhisperTokenIDConverter):
    tokens = whisper_token_id_converter.ids2tokens(
        [17155, 11, 220, 83, 378, 320, 311, 5503, 307, 1481, 13, 8239, 485]
    )

    assert tokens == [
        "Hi",
        ",",
        "Ġ",
        "t",
        "od",
        "ay",
        "'s",
        "Ġweather",
        "Ġis",
        "Ġnice",
        ".",
        "ĠHmm",
        "...",
    ]


def test_tokens2ids(whisper_token_id_converter: OpenAIWhisperTokenIDConverter):
    ids = whisper_token_id_converter.tokens2ids(
        [
            "Hi",
            ",",
            "Ġ",
            "t",
            "od",
            "ay",
            "'s",
            "Ġweather",
            "Ġis",
            "Ġnice",
            ".",
            "ĠHmm",
            "...",
        ]
    )

    assert ids == [
        50259,
        50359,
        50363,
        17155,
        11,
        220,
        83,
        378,
        320,
        311,
        5503,
        307,
        1481,
        13,
        8239,
        485,
    ]
