"""A demo notebook from a declaration: cells, rules of the notebook repo, CLI."""

import json

import pytest

from espnet3.api.inference import Field, InferenceAPI
from espnet3.publication import notebook
from espnet3.publication.usage import inference_class


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


class Say(InferenceAPI):
    """Text in, audio out."""

    inputs = (Field("text", "text", "Text"),)
    outputs = (Field("audio", "audio", "Speech"),)

    @classmethod
    def from_pretrained(cls, tag_or_dir, *, device="auto", **kwargs):
        return cls()

    def run(self, text):
        return {"audio": text}


def _sources(page, cell_type=None):
    return [c["source"] for c in page["cells"] if cell_type in (None, c["cell_type"])]


def test_the_page_is_a_valid_notebook_and_follows_the_demo_rules():
    nbformat = pytest.importorskip("nbformat")
    page = notebook.demo_notebook(
        Describe,
        "espnet/some_pack",
        name="asr_demo",
        title="Describe speech",
        espnet_version="202610.post2",
    )

    nbformat.validate(nbformat.from_dict(page))
    text = "\n".join(_sources(page))
    # what espnet/notebook's tools/check_layout.py holds a demo to
    assert 'espnet==202610.post2"' in text
    assert "workflows/asr_demo.yml/badge.svg" in text
    assert "git+https://github.com/espnet/espnet" not in text
    assert (
        "colab.research.google.com/github/espnet/notebook/blob/master/Demos/"
        "asr_demo.ipynb" in text
    )


def test_the_cells_come_from_the_declaration():
    page = notebook.demo_notebook(
        Describe,
        "espnet/some_pack",
        name="asr_demo",
        title="Describe speech",
        espnet_version="202610.post2",
        space="https://huggingface.co/spaces/espnet/x",
    )
    code = _sources(page, "code")

    assert code[0] == '%pip install -q "espnet==202610.post2" espnet_model_zoo'
    assert code[1].startswith(f"!wget -q -O speech.wav {notebook.SAMPLE_AUDIO}\n")
    assert 'display(Audio("speech.wav"))' in code[1]
    assert 'model = load("espnet/some_pack")\nresult = model("speech.wav")' in code[2]
    assert "# optional inputs: hint (text)" in code[2]
    assert code[3].splitlines() == [
        "from IPython.display import Audio, display",
        "",
        'print(result["text"])',
        'for segment in result["segments"]:',
        "    print(f\"{segment['start']:7.2f} {segment['end']:7.2f}  "
        "{segment['text']}\")",
        'display(Audio(result["audio"].array, rate=result["audio"].rate))',
    ]
    assert code[4].startswith("# several at once")
    last = _sources(page, "markdown")[-1]
    assert "`espnet run espnet/some_pack speech=speech.wav`" in last
    assert "[Space](https://huggingface.co/spaces/espnet/x)" in last
    assert "https://huggingface.co/espnet/some_pack" in last
    assert (
        "Audio and an optional hint in; text, spans and audio out."
        in page["cells"][0]["source"]
    )


def test_a_text_input_is_given_and_no_recording_is_fetched():
    page = notebook.demo_notebook(
        Say,
        "espnet/some_tts",
        name="tts_demo",
        title="Say it",
        inputs={"text": "Hello from ESPnet"},
        espnet_version="202610.post2",
    )
    code = _sources(page, "code")

    assert not any("wget" in c for c in code)
    assert 'result = model("Hello from ESPnet")' in code[1]
    assert (
        'model.batch([{"text": "Hello from ESPnet"}, {"text": "Hello from ESPnet"}])'
        in code[3]
    )
    assert (
        "`espnet run espnet/some_tts text=Hello from ESPnet`"
        in _sources(page, "markdown")[-1]
    )


def test_an_optional_input_named_makes_the_page_about_it():
    """One notebook per task: st_demo passes task=st_deu on the asr model."""
    page = notebook.demo_notebook(
        inference_class("s2t"),
        "espnet/owsm_v4_base_102M",
        system="s2t",
        name="st_demo",
        title="Speech translation with OWSM v4",
        inputs={"language": "eng", "task": "st_deu"},
        espnet_version="202610.post2",
    )
    code = _sources(page, "code")

    assert 'model = load("espnet/owsm_v4_base_102M", system="s2t")' in code[2]
    assert 'result = model("speech.wav", language="eng", task="st_deu")' in code[2]
    assert "# optional inputs: previous_text (text)" in code[2]
    item = '{"speech": "speech.wav", "language": "eng", "task": "st_deu"}'
    assert f"model.batch([{item}, {item}])" in code[4]
    assert (
        "`espnet run espnet/owsm_v4_base_102M --system s2t speech=speech.wav "
        "language=eng task=st_deu`"
    ) in _sources(page, "markdown")[-1]


def test_no_version_means_no_pin(monkeypatch):
    import espnet

    monkeypatch.setattr(espnet, "__version__", "unknown (running from a source tree)")
    page = notebook.demo_notebook(
        Say, "x", name="tts_demo", title="t", inputs={"text": "a"}
    )
    assert _sources(page, "code")[0] == "%pip install -q espnet espnet_model_zoo"


def test_main_writes_the_file(tmp_path, capsys):
    out = tmp_path / "Demos" / "pr_demo.ipynb"
    argv = [
        "espnet/powsm_ctc",
        "--system",
        "s2t",
        "--title",
        "Phones with POWSM",
        "--input",
        "task=pr",
        "--espnet-version",
        "202610.post2",
        "-o",
        str(out),
    ]
    assert notebook.main(argv) == 0

    assert capsys.readouterr().out.strip() == str(out)
    page = json.loads(out.read_text())
    assert page["nbformat"] == 4
    text = "\n".join(_sources(page))
    assert 'result = model("speech.wav", task="pr")' in text
    assert "workflows/pr_demo.yml/badge.svg" in text


def test_main_reads_the_system_from_the_bundle_when_none_is_named(
    tmp_path, monkeypatch, capsys
):
    """No weights are built for it: meta.yaml names the system."""
    import espnet3.api.inference.loading as loading

    monkeypatch.setattr(loading, "locate_pack", lambda tag: tmp_path)
    monkeypatch.setattr(loading, "read_meta", lambda path: {"system": "s2t"})
    out = tmp_path / "asr_demo.ipynb"
    assert notebook.main(["espnet/some_pack", "--title", "t", "-o", str(out)]) == 0

    text = "\n".join(_sources(json.loads(out.read_text())))
    assert 'model = load("espnet/some_pack")\nresult = model("speech.wav")' in text


def test_main_asks_for_the_system_a_bundle_does_not_name(tmp_path, monkeypatch, capsys):
    import espnet3.api.inference.loading as loading

    monkeypatch.setattr(loading, "locate_pack", lambda tag: tmp_path)
    monkeypatch.setattr(loading, "read_meta", lambda path: {})
    with pytest.raises(SystemExit) as e:
        notebook.main(["espnet/some_pack", "--title", "t", "-o", str(tmp_path / "a")])
    assert e.value.code == 2
    assert "names no system in its meta.yaml; pass --system" in capsys.readouterr().err


def test_main_refuses_an_input_without_a_value(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        notebook.main(
            [
                "x",
                "--system",
                "s2t",
                "--title",
                "t",
                "-o",
                str(tmp_path / "a.ipynb"),
                "--input",
                "task",
            ]
        )
    assert e.value.code == 2
    assert "give NAME=VALUE" in capsys.readouterr().err
