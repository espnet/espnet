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
    # 50257 base + 107 specials + 1501 timestamps. This used to read 51867,
    # which only held while an earlier test had grown the shared tokenizer
    # through the added_tokens_txt path; that path now copies first.
    assert id_converter.get_num_vocabulary_size() == 51865


def test_init_translation():
    id_converter = OpenAIWhisperTokenIDConverter(
        "whisper_multilingual", "zh", task="translate"
    )
    assert id_converter.get_num_vocabulary_size() == 51865


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


def test_token2id(whisper_token_id_converter: OpenAIWhisperTokenIDConverter):
    token2id = whisper_token_id_converter.token2id
    assert token2id["<|startoftranscript|>"] == 50258
    assert token2id["<|en|>"] == 50259
    assert token2id["<|transcribe|>"] == 50359
    assert token2id["<|notimestamps|>"] == 50363
    first = token2id["<|0.00|>"]
    last = token2id["<|30.00|>"]
    assert token2id["<|0.02|>"] == first + 1
    assert last - first == 1500


@pytest.mark.parametrize("sym", ["????", "<sc>"])
def test_token2id_speaker_change_symbol(sym):
    converter = OpenAIWhisperTokenIDConverter(
        "whisper_multilingual", sot=True, speaker_change_symbol=sym
    )
    if sym == "????":
        # Already a single Whisper BPE token: sot must reuse that row, not
        # mint a new id past the vocabulary.
        assert converter.token2id[sym] == 25629
    else:
        # A symbol Whisper lacks lands above the last timestamp row (51864).
        assert converter.token2id[sym] > 51864


def test_ids2tokens_keep_special_tokens():
    dropped = OpenAIWhisperTokenIDConverter("whisper_multilingual")
    kept = OpenAIWhisperTokenIDConverter(
        "whisper_multilingual", keep_special_tokens=True
    )
    ids = kept.tokenizer.tokenizer.convert_tokens_to_ids(["<|0.00|>", "Hi", "<|0.02|>"])
    assert dropped.ids2tokens(ids) == ["Hi"]
    assert kept.ids2tokens(ids) == ["<|0.00|>", "Hi", "<|0.02|>"]
