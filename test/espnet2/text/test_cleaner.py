import pytest

from espnet2.text.cleaner import TextCleaner


@pytest.fixture(params=["tacotron", "jaconv"])
def cleaner(request):
    return TextCleaner(request.param)


def test_repr(cleaner):
    print(cleaner)


def test_call(cleaner):
    if cleaner.cleaner_types[0] == "tacotron":
        assert (
            cleaner("(Hello-World);   &  jr. & dr.")
            == "HELLO WORLD, AND JUNIOR AND DOCTOR"
        )
    elif cleaner.cleaner_types[0] == "jaconv":
        assert cleaner("ティロ･フィナ〜レ") == "ティロ・フィナーレ"
