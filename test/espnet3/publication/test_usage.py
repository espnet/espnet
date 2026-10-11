"""Usage text from a system's declaration, no weights loaded."""

import pytest

from espnet3.api.inference import Field, InferenceAPI
from espnet3.publication.usage import inference_class, python_usage, usage_section


class Describe(InferenceAPI):
    """Audio and an optional hint in; text, spans and audio out."""

    inputs = (
        Field("speech", "audio", "Speech"),
        Field("hint", "text", "Hint", optional=True),
    )
    outputs = (
        Field("text", "text", "Description"),
        Field("segments", "segments", "Spans"),
        Field("audio", "audio", "Rendered"),
    )

    @classmethod
    def from_pretrained(cls, tag_or_dir, *, device="auto", **kwargs):
        return cls()

    def run(self, speech, hint=None):
        return {"text": "", "segments": [], "audio": speech}


def test_the_usage_follows_the_declaration():
    text = python_usage(Describe, "espnet/some_pack")
    assert text.splitlines() == [
        "from espnet3.api.inference import load",
        "",
        'model = load("espnet/some_pack")',
        'result = model("speech.wav")',
        "# optional inputs: hint (text)",
        '# result = model("speech.wav", hint="...")',
        'result["text"]      # Description: str',
        "result[\"segments\"]  # Spans: [{'text', 'start', 'end'}, ...] in seconds",
        'result["audio"]     # Rendered: an Audio: .array (samples) at .rate',
        "",
        "# several at once, and pieces as the model produces them",
        'model.batch([{"speech": "speech.wav"}, {"speech": "speech.wav"}])',
        'for piece in model.stream([{"speech": "speech.wav"}]):',
        "    print(piece)",
    ]


def test_a_checkpoint_without_a_bundle_names_its_system():
    """load(tag, system=...) is what a published ESPnet2 checkpoint needs."""
    text = python_usage(
        inference_class("s2t"), "espnet/owsm_v4_base_102M", system="s2t"
    )
    assert 'model = load("espnet/owsm_v4_base_102M", system="s2t")' in text
    assert (
        "# optional inputs: language (text), task (text), previous_text (text)" in text
    )
    assert 'result["segments"]' in text


def test_inference_class_follows_the_aliases_load_follows():
    from espnet3.systems.esp2_asr.inference import Inference as Asr
    from espnet3.systems.esp2_s2t.inference import Inference as S2t

    assert inference_class("asr") is Asr and inference_class("esp2_asr") is Asr
    assert inference_class("s2t") is S2t
    with pytest.raises(ImportError):
        inference_class("no_such_system")


def test_usage_section_is_a_fenced_block_under_a_heading():
    section = usage_section(Describe, "espnet/some_pack")
    assert section.startswith("## Usage\n\n```python\n")
    assert section.endswith("\n```")
