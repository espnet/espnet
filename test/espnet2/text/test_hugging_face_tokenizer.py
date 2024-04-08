import pytest

from espnet2.text.hugging_face_tokenizer import HuggingFaceTokenizer


@pytest.fixture(params=["sshleifer/tiny-mbart"])
def hugging_face_tokenizer(request):
    return HuggingFaceTokenizer(request.param)


@pytest.mark.execution_timeout(50)
def test_repr(hugging_face_tokenizer: HuggingFaceTokenizer):
    print(hugging_face_tokenizer)


@pytest.mark.execution_timeout(50)
def test_text2tokens(hugging_face_tokenizer: HuggingFaceTokenizer):
    assert hugging_face_tokenizer.text2tokens("Hello World!! Ummm") == [
        "▁Hello",
        "▁World",
        "!!",
        "▁Um",
        "mm",
    ]


@pytest.mark.execution_timeout(50)
def test_tokens2text(hugging_face_tokenizer: HuggingFaceTokenizer):
    assert (
        hugging_face_tokenizer.tokens2text(["▁Hello", "▁World", "!!", "▁Um", "mm"])
        == "Hello World!! Ummm"
    )
