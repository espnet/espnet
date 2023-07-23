import pytest

from espnet2.text.hugging_face_token_id_converter import HuggingFaceTokenIDConverter


@pytest.fixture(params=["EleutherAI/pythia-410m-deduped"])
def hugging_face_token_id_converter(request):
    return HuggingFaceTokenIDConverter(request.param)


def test_init_pythia():
    id_converter = HuggingFaceTokenIDConverter("EleutherAI/pythia-410m-deduped")
    assert id_converter.get_num_vocabulary_size() == 50254


def test_ids2tokens(hugging_face_token_id_converter: HuggingFaceTokenIDConverter):
    tokens = hugging_face_token_id_converter.ids2tokens(
        [30377, 16580, 18128, 16, 4789, 36005, 14, 30889, 78, 14, 4861, 484, 264]
    )

    assert tokens == [
        "Ele",
        "uther",
        "AI",
        "/",
        "py",
        "thia",
        "-",
        "410",
        "m",
        "-",
        "ded",
        "up",
        "ed",
    ]


def test_tokens2ids(hugging_face_token_id_converter: HuggingFaceTokenIDConverter):
    ids = hugging_face_token_id_converter.tokens2ids(
        [
            "Ele",
            "uther",
            "AI",
            "/",
            "py",
            "thia",
            "-",
            "410",
            "m",
            "-",
            "ded",
            "up",
            "ed",
        ]
    )

    assert ids == [
        30377,
        16580,
        18128,
        16,
        4789,
        36005,
        14,
        30889,
        78,
        14,
        4861,
        484,
        264,
    ]
