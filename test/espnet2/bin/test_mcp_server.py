import asyncio
import importlib
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile
import torch
from mcp.server.mcpserver.exceptions import ToolError

from espnet2.bin import mcp_server


def call(server, name, **arguments):
    result = asyncio.run(server.call_tool(name, arguments))
    assert not result.is_error, result
    return result.content[0].text


@pytest.fixture()
def wav(tmp_path):
    path = tmp_path / "in.wav"
    soundfile.write(str(path), np.zeros(8000, dtype="float32"), 8000)
    return path


def test_tools_are_listed_with_their_docs():
    tools = {t.name: t for t in asyncio.run(mcp_server.build_server().list_tools())}
    # the names are the `espnet` subcommands, so that a transcript of an
    # agent's work and a terminal history read the same
    assert set(tools) == {
        "transcribe",
        "translate",
        "phonemize",
        "align",
        "synthesize",
        "enhance",
        "describe",
        "render",
    }
    assert tools["transcribe"].input_schema["required"] == ["audio_path"]
    assert set(tools["transcribe"].input_schema["properties"]) == {
        "audio_path",
        "language",
    }
    assert tools["translate"].input_schema["required"] == ["audio_path", "to"]
    assert tools["align"].input_schema["required"] == ["audio_path", "utterances"]
    assert tools["synthesize"].input_schema["required"] == ["text", "output_path"]
    assert "ISO 639-3" in tools["transcribe"].description


class FakeASR:
    """An esp2_s2t Inference stand-in: language and task by name, text out.

    It refuses what the real one refuses, with the real one's sentences, so
    that the server's relay of them is what is tested here.
    """

    languages = ("eng", "deu", "jpn")
    targets = ("deu", "eng")

    def __init__(self):
        self.calls = []

    def __call__(self, audio, **prompt):
        language, task = prompt.get("language"), prompt["task"]
        if language is not None and language not in self.languages:
            raise ValueError(
                f"language {language!r}: this model has no <{language}>; pass "
                "ISO 639-3, three letters: eng, deu, jpn, zho, fra, spa ..."
            )
        if task.startswith("st_") and task[3:] not in self.targets:
            raise ValueError(
                f"task {task!r}: this model has no <{task}>; it takes asr, or "
                f"st_<ISO 639-3> to translate, into {', '.join(self.targets)}"
            )
        self.calls.append((audio, prompt))
        return {"text": "hallo welt" if task == "st_deu" else "hello world"}


def test_transcribe_maps_codes_to_symbols(monkeypatch, wav):
    fake = FakeASR()
    monkeypatch.setattr(mcp_server, "_asr", lambda: fake)
    server = mcp_server.build_server()
    assert call(server, "transcribe", audio_path=str(wav)) == "hello world"
    assert (
        call(server, "translate", audio_path=str(wav), language="eng", to="deu")
        == "hallo welt"
    )
    # by name; no language means the model works it out
    assert fake.calls == [
        (str(wav), {"task": "asr"}),
        (str(wav), {"task": "st_deu", "language": "eng"}),
    ]


def test_translation_says_when_neither_side_is_english(monkeypatch, wav):
    """OWSM's translation training pairs English with the other language."""
    fake = FakeASR()
    monkeypatch.setattr(mcp_server, "_asr", lambda: fake)
    server = mcp_server.build_server()

    with pytest.warns(UserWarning, match="English on neither side"):
        call(server, "translate", audio_path=str(wav), language="jpn", to="deu")

    # a direction the model was trained on, and one the server cannot know
    # the source of, are not warned about
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        call(server, "translate", audio_path=str(wav), language="jpn", to="eng")
        call(server, "translate", audio_path=str(wav), language="eng", to="deu")
        call(server, "translate", audio_path=str(wav), to="deu")


def test_transcribe_rejects_what_the_model_cannot_do(monkeypatch, wav, tmp_path):
    monkeypatch.setattr(mcp_server, "_asr", FakeASR)
    server = mcp_server.build_server()
    with pytest.raises(ToolError, match="ISO 639-3"):
        asyncio.run(
            server.call_tool("transcribe", {"audio_path": str(wav), "language": "en"})
        )
    # fra is ISO 639-3, so the answer is the targets the model has
    with pytest.raises(ToolError, match="no <st_fra>.*into deu, eng"):
        asyncio.run(
            server.call_tool("translate", {"audio_path": str(wav), "to": "fra"})
        )
    with pytest.raises(ToolError, match="No such audio file"):
        asyncio.run(
            server.call_tool("transcribe", {"audio_path": str(tmp_path / "no.wav")})
        )


class FakePOWSM(FakeASR):
    """POWSM writes each phone between slashes; the server takes them out."""

    languages = ("eng",)

    def __call__(self, audio, **prompt):
        super().__call__(audio, **prompt)
        return {"text": "/ð//ə/ /s//eɪ//l/"}


def test_phonemize_returns_one_token_a_phone(monkeypatch, wav):
    fake = FakePOWSM()
    monkeypatch.setattr(mcp_server, "_pr", lambda: fake)
    server = mcp_server.build_server()

    assert call(server, "phonemize", audio_path=str(wav)) == "ð ə s eɪ l"

    # phones by name; no language means the model works it out
    assert fake.calls == [(str(wav), {"task": "pr"})]


def test_phonemize_takes_a_language_the_model_has(monkeypatch, wav):
    fake = FakePOWSM()
    monkeypatch.setattr(mcp_server, "_pr", lambda: fake)
    server = mcp_server.build_server()

    call(server, "phonemize", audio_path=str(wav), language="eng")
    assert fake.calls[-1][1] == {"task": "pr", "language": "eng"}

    with pytest.raises(ToolError, match="ISO 639-3"):
        asyncio.run(
            server.call_tool(
                "phonemize", {"audio_path": str(wav), "language": "english"}
            )
        )


class FakeAligner:
    """ForcedAligner returns a segment an utterance, with a probability."""

    def __call__(self, speech, utterances):
        self.seen = (speech, list(utterances))
        return [
            SimpleNamespace(text=text, start=i * 1.0, end=i + 0.5, score=0.25)
            for i, text in enumerate(utterances)
        ]


def test_align_prints_a_line_an_utterance(monkeypatch, wav):
    fake = FakeAligner()
    monkeypatch.setattr(mcp_server, "_aligner", lambda: fake)
    server = mcp_server.build_server()

    printed = call(server, "align", audio_path=str(wav), utterances=["one", "two"])

    assert printed.split("\n") == [
        "0.00\t0.50\t0.2500\tone",
        "1.00\t1.50\t0.2500\ttwo",
    ]
    # the path, not the samples: the aligner reads the file the way its own
    # model needs it read
    assert fake.seen == (str(wav), ["one", "two"])


def test_align_reports_text_that_cannot_fit(monkeypatch, wav):
    class TooMuch:
        def __call__(self, speech, utterances):
            raise ValueError("120 tokens to align against 40 frames")

    monkeypatch.setattr(mcp_server, "_aligner", TooMuch)
    server = mcp_server.build_server()

    with pytest.raises(ToolError, match="120 tokens"):
        asyncio.run(
            server.call_tool("align", {"audio_path": str(wav), "utterances": ["one"]})
        )


def test_align_without_utterances_says_so(monkeypatch, wav):
    monkeypatch.setattr(mcp_server, "_aligner", FakeAligner)
    server = mcp_server.build_server()

    with pytest.raises(ToolError, match="utterances"):
        asyncio.run(
            server.call_tool("align", {"audio_path": str(wav), "utterances": [" "]})
        )


def test_synthesize_writes_a_wav_at_the_model_rate(monkeypatch, tmp_path):
    class TTS:
        fs = 22050

        def __call__(self, text):
            assert text == "hello"
            return {"wav": torch.zeros(2205)}

    monkeypatch.setattr(mcp_server, "_tts", TTS)
    out = tmp_path / "nested" / "out.wav"
    written = call(
        mcp_server.build_server(), "synthesize", text="hello", output_path=str(out)
    )
    assert written == str(out.resolve())
    data, fs = soundfile.read(str(out))
    assert fs == 22050 and len(data) == 2205


def test_synthesize_refuses_a_model_without_a_rate(monkeypatch, tmp_path):
    class TTS:
        fs = None

        def __call__(self, text):
            raise AssertionError("must not synthesize without a rate to write at")

    monkeypatch.setattr(mcp_server, "_tts", TTS)
    with pytest.raises(ToolError, match="sampling rate"):
        asyncio.run(
            mcp_server.build_server().call_tool(
                "synthesize", {"text": "x", "output_path": str(tmp_path / "o.wav")}
            )
        )


def test_enhance_resamples_and_writes(monkeypatch, wav, tmp_path):
    # Pin the rate: a runner with ESPNET_MCP_ENH_FS set would otherwise decide
    # whether the 8 kHz file gets resampled at all.
    monkeypatch.setattr(mcp_server, "ENH_FS", 16000)
    seen = {}

    class ENH:
        def __call__(self, speech_mix, fs):
            seen["shape"], seen["fs"] = speech_mix.shape, fs
            return [torch.from_numpy(speech_mix)]

    monkeypatch.setattr(mcp_server, "_enh", ENH)
    out = tmp_path / "clean.wav"
    call(
        mcp_server.build_server(), "enhance", audio_path=str(wav), output_path=str(out)
    )
    # 1 s at 8 kHz in, resampled to the model's 16 kHz, one channel.
    assert seen["fs"] == mcp_server.ENH_FS == 16000
    assert seen["shape"] == (1, 16000)
    data, fs = soundfile.read(str(out))
    assert fs == 16000 and len(data) == 16000


def test_the_speech_to_text_loaders_go_through_espnet3(monkeypatch):
    """_s2t loads the tag as the s2t system, on the device settled for this machine."""
    import torch

    import espnet3.api.inference as api

    seen = []
    monkeypatch.setattr(
        api, "load", lambda tag, **kw: seen.append((tag, kw)) or "model"
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mcp_server, "DEVICE", "auto")

    assert mcp_server._s2t("espnet/some_owsm") == "model"
    assert seen == [("espnet/some_owsm", {"system": "s2t", "device": "cpu"})]


def test_build_server_names_the_missing_package(monkeypatch):
    monkeypatch.setattr(mcp_server, "MCPServer", None)
    with pytest.raises(ImportError, match=r"espnet\[mcp\]"):
        mcp_server.build_server()


def test_enhancement_rate_is_validated_at_import(monkeypatch):
    try:
        with monkeypatch.context() as env:
            env.setenv("ESPNET_MCP_ENH_FS", "0")
            with pytest.raises(ValueError, match="positive integer"):
                importlib.reload(mcp_server)
            env.delenv("ESPNET_MCP_ENH_FS")
            importlib.reload(mcp_server)
            assert mcp_server.ENH_FS == 16000
    finally:
        # Reload once more with the runner's own environment restored, so the
        # module globals other tests read match it again.
        importlib.reload(mcp_server)


class _FakeBagpiper:
    """A served Bagpiper, without a server."""

    def __init__(self, text="a bell rings", audio=None):
        self.text, self.audio = text, audio
        self.asked = []

    def describe(self, audio, prompt=None, **kwargs):
        self.asked.append((audio, prompt))
        return self.text

    def render(self, scene, **kwargs):
        self.asked.append(scene)
        return self.audio, self.text


def _bagpiper(monkeypatch, model):
    monkeypatch.setattr(mcp_server, "_bagpiper", lambda: model)
    return model


def test_describe_answers_about_the_audio(monkeypatch, wav):
    model = _bagpiper(monkeypatch, _FakeBagpiper())

    answer = call(mcp_server.build_server(), "describe", audio_path=str(wav))

    assert answer == "a bell rings"
    assert model.asked == [(str(wav), "What sound is in this audio?")]


def test_describe_refuses_a_missing_file(monkeypatch):
    _bagpiper(monkeypatch, _FakeBagpiper())

    with pytest.raises(ToolError):
        mcp_server.describe("/nowhere/missing.wav")


def test_render_writes_the_wav_it_was_given(monkeypatch, tmp_path):
    from test.espnet2.bin.test_speechlm_inference import wav_bytes

    _bagpiper(monkeypatch, _FakeBagpiper(audio=wav_bytes()))
    out = tmp_path / "made" / "bell.wav"

    written = call(
        mcp_server.build_server(),
        "render",
        scene="A bell rings twice.",
        output_path=str(out),
    )

    assert written == str(out.resolve())
    assert out.read_bytes() == wav_bytes()


def test_render_reports_a_text_only_answer(monkeypatch, tmp_path):
    _bagpiper(monkeypatch, _FakeBagpiper(text="I would render a bell."))

    with pytest.raises(ToolError) as raised:
        mcp_server.render("read this aloud: hello", str(tmp_path / "x.wav"))

    assert "described scene" in str(raised.value)
    assert not (tmp_path / "x.wav").exists()


def test_a_server_that_is_not_running_is_reported_as_such(monkeypatch):
    """The other tools download; this one has an address, and says so."""
    monkeypatch.setattr(mcp_server, "BAGPIPER_URL", "http://127.0.0.1:9/v1")
    mcp_server._bagpiper.cache_clear()

    with pytest.raises(RuntimeError) as raised:
        mcp_server._bagpiper().describe(__file__)

    assert "docker run" in str(raised.value) or "README" in str(raised.value)
    mcp_server._bagpiper.cache_clear()
