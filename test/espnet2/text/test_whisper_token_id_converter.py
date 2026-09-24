import pytest

from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter

pytest.importorskip("whisper")


@pytest.fixture(params=["whisper_multilingual"])
def whisper_token_id_converter(request):
    return OpenAIWhisperTokenIDConverter(request.param)


def test_init_model_invalid():
    with pytest.raises(ValueError):
        OpenAIWhisperTokenIDConverter("whisper_aaa", "en", task="transcribe")


def test_init_lang_invalid():
    with pytest.raises(ValueError):
        OpenAIWhisperTokenIDConverter("whisper_multilingual", "abc", task="transcribe")


def test_init_task_invalid():
    with pytest.raises(ValueError):
        OpenAIWhisperTokenIDConverter(
            "whisper_multilingual", "zh", task="transcribe_abc"
        )


def test_init_en():
    id_converter = OpenAIWhisperTokenIDConverter("whisper_en", "en", task="transcribe")
    assert id_converter.get_num_vocabulary_size() == 51864


def test_init_multilingual():
    id_converter = OpenAIWhisperTokenIDConverter(
        "whisper_multilingual", "zh", task="transcribe"
    )
    assert id_converter.get_num_vocabulary_size() == 51867


def test_init_translation():
    id_converter = OpenAIWhisperTokenIDConverter(
        "whisper_multilingual", "zh", task="translate"
    )
    assert id_converter.get_num_vocabulary_size() == 51867


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

    assert ids[0] == 50259


def test_tokens2ids_add_tokens(tmp_path):
    tknlist_path = tmp_path / "tmp_token_list/add_token_list.txt"
    tknlist_path.parent.mkdir()
    tknlist_path.touch()
    with open(tknlist_path, "w") as f:
        f.write("command:yes\n")
    _ = OpenAIWhisperTokenIDConverter(
        "whisper_multilingual", added_tokens_txt=str(tknlist_path)
    )
