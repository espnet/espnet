import pytest

from espnet2.text.whisper_tokenizer import OpenAIWhisperTokenizer

pytest.importorskip("whisper")


@pytest.fixture(params=["whisper_multilingual"])
def whisper_tokenizer(request):
    return OpenAIWhisperTokenizer(request.param)


def test_init_en():
    tokenizer = OpenAIWhisperTokenizer("whisper_en", "en", "transcribe")
    assert tokenizer.tokenizer.tokenizer.vocab_size == 50257


def test_init_multilingual():
    tokenizer = OpenAIWhisperTokenizer("whisper_multilingual", "zh", "transcribe")
    assert tokenizer.tokenizer.tokenizer.vocab_size == 50257


def test_init_translation():
    tokenizer = OpenAIWhisperTokenizer("whisper_multilingual", "zh", "translate")
    assert tokenizer.tokenizer.tokenizer.vocab_size == 50257


def test_init_model_invalid():
    with pytest.raises(ValueError):
        OpenAIWhisperTokenizer("whisper_aaa", "en", "transcribe")


def test_init_lang_invalid():
    with pytest.raises(ValueError):
        OpenAIWhisperTokenizer("whisper_multilingual", "aaa", "transcribe")


def test_init_task_invalid():
    with pytest.raises(ValueError):
        OpenAIWhisperTokenizer("whisper_multilingual", "zh", "transcribe_aaa")


def test_repr(whisper_tokenizer: OpenAIWhisperTokenizer):
    print(whisper_tokenizer)


def test_tokenization_consistency(whisper_tokenizer: OpenAIWhisperTokenizer):
    s = "Hi, today's weather is nice. Hmm..."

    assert s == whisper_tokenizer.tokens2text(whisper_tokenizer.text2tokens(s))


def test_tokenization_add_tokens(tmp_path):
    tknlist_path = tmp_path / "tmp_token_list/add_token_list.txt"
    tknlist_path.parent.mkdir()
    tknlist_path.touch()
    with open(tknlist_path, "w") as f:
        f.write("command:yes\n")
    _ = OpenAIWhisperTokenizer(
        "whisper_multilingual", added_tokens_txt=str(tknlist_path)
    )
