"""The read path's prompt transform.

``PromptFromTags`` is what replaced the recipe's own ``Speech2TextOWSM``: the
language and task reach the model as dataset columns now, so they have to
come out of the reference's tags exactly as that class read them.
"""

from __future__ import annotations

import pytest

from egs3.owsm_v4.esp2_s2t.dataset.dataset import PromptFromTags


@pytest.mark.parametrize(
    "text, language, task",
    [
        ("<eng><asr> hello world", "eng", "asr"),
        ("<deu><st_eng> hallo welt", "deu", "st_eng"),
        # timestamps follow the prefix in a v4 reference
        ("<eng><asr><0.00> hello<1.50>", "eng", "asr"),
        (" <eng> <asr> spaced out", "eng", "asr"),
    ],
)
def test_the_tags_become_columns_without_their_brackets(text, language, task):
    assert PromptFromTags()({"text": text}) == {
        "text": text,
        "language": language,
        "task": task,
    }


@pytest.mark.parametrize("text", ["plain text", "", None, "<eng> only one tag"])
def test_an_untagged_reference_prompts_nothing(text):
    """The model then works the language out itself, as it does for a bare wav."""
    assert PromptFromTags()({"text": text}) == {"text": text}


def test_the_other_columns_are_left_alone():
    sample = {"speech": [0.0], "text": "<eng><asr> hi", "text_prev": "<na>"}
    out = PromptFromTags()(sample)
    assert out["speech"] is sample["speech"]
    assert out["text_prev"] == "<na>"
    assert sample == {"speech": [0.0], "text": "<eng><asr> hi", "text_prev": "<na>"}
