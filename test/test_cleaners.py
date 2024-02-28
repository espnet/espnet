# coding: utf-8

import pytest
from typing import Collection
from espnet2.text.cleaner import TextCleaner

ENGLISH_DIRTY_TEXT = "(Hello-$10);  'i am Joe's jr. & dr.',ëé\t\nßæãåūúìîóœø!!"


@pytest.mark.parametrize(
    "cleaner_types, expected_clean_text", [(
        "tacotron",
        "HELLO TEN DOLLARS, 'I AM JOE'S JUNIOR AND DOCTOR',EE SSAEAAUUIIOOEO!!"
    ), (
        "mfa_english",
        "hello ten dollars, i am joe's junior and doctor, ee ssaeaauuiiooeo!!"
    ), (
        ["tacotron", "mfa_english"],
        "hello ten dollars, i am joe's junior and doctor, ee ssaeaauuiiooeo!!"
    ),
        # Commented out for now due to whisper not installed on some CI settings
        # (
        #     "whisper_en",
        #     "i am joe is junior doctor ee ssaeaauuiiooeo"
        # ), (
        #     "whisper_basic",
        #     " i am joe s jr dr ëé ßæãåūúìîóœø "
        # ),
    ]
)
def test_english(cleaner_types: Collection[str], expected_clean_text: str):
    cleaner = TextCleaner(cleaner_types)
    assert cleaner(ENGLISH_DIRTY_TEXT) == expected_clean_text

# Test for Japanese, Vietnamese and Korean?
