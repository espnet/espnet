import sys

import pytest

from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter

pytest.importorskip("whisper")

is_python_3_8_plus = sys.version_info >= (3, 8)


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
@pytest.fixture(params=["whisper_multilingual"])
def whisper_token_id_converter(request):
    return OpenAIWhisperTokenIDConverter(request.param)


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
def test_init_invalid():
    with pytest.raises(ValueError):
        OpenAIWhisperTokenIDConverter("whisper_aaa")


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
def test_init_en():
    id_converter = OpenAIWhisperTokenIDConverter("whisper_en")
    assert id_converter.get_num_vocabulary_size() == 50363


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
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


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
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
