import sys

import pytest

from espnet2.text.whisper_tokenizer import OpenAIWhisperTokenizer

pytest.importorskip("whisper")

is_python_3_8_plus = sys.version_info >= (3, 8)


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
@pytest.fixture(params=["whisper_multilingual"])
def whisper_tokenizer(request):
    return OpenAIWhisperTokenizer(request.param)


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
def test_init_en():
    tokenizer = OpenAIWhisperTokenizer("whisper_en", "en", "transcribe")
    assert tokenizer.tokenizer.tokenizer.vocab_size == 50257


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
def test_init_multilingual():
    tokenizer = OpenAIWhisperTokenizer("whisper_multilingual", "zh", "transcribe")
    assert tokenizer.tokenizer.tokenizer.vocab_size == 50257


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
def test_init_translation():
    tokenizer = OpenAIWhisperTokenizer("whisper_multilingual", "zh", "translate")
    assert tokenizer.tokenizer.tokenizer.vocab_size == 50257


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
def test_init_model_invalid():
    with pytest.raises(ValueError):
        OpenAIWhisperTokenizer("whisper_aaa", "en", "transcribe")


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
def test_init_lang_invalid():
    with pytest.raises(ValueError):
        OpenAIWhisperTokenizer("whisper_multilingual", "aaa", "transcribe")


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
def test_init_task_invalid():
    with pytest.raises(ValueError):
        OpenAIWhisperTokenizer("whisper_multilingual", "zh", "transcribe_aaa")


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
def test_repr(whisper_tokenizer: OpenAIWhisperTokenizer):
    print(whisper_tokenizer)


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
def test_tokenization_consistency(whisper_tokenizer: OpenAIWhisperTokenizer):
    s = "Hi, today's weather is nice. Hmm..."

    assert s == whisper_tokenizer.tokens2text(whisper_tokenizer.text2tokens(s))


@pytest.mark.skipif(
    not is_python_3_8_plus, reason="whisper not supported on python<3.8"
)
def test_tokenization_add_tokens(tmp_path):
    tknlist_path = tmp_path / "tmp_token_list/add_token_list.txt"
    tknlist_path.parent.mkdir()
    tknlist_path.touch()
    with open(tknlist_path, "w") as f:
        f.write("command:yes\n")
    _ = OpenAIWhisperTokenizer(
        "whisper_multilingual", added_tokens_txt=str(tknlist_path)
    )
