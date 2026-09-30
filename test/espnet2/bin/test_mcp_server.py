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
    def __init__(self):
        self.calls = []
        self.s2t_model = SimpleNamespace(
            token_list=[
                "<nolang>",
                "<eng>",
                "<deu>",
                "<jpn>",
                "<asr>",
                "<st_deu>",
                "<st_eng>",
            ]
        )

    def decode_long(self, speech, lang_sym, task_sym):
        self.calls.append((speech, lang_sym, task_sym))
        # decode_long returns one (start, end, text) per segment; the two
        # segments are joined by the server, so a single-word return here
        # would not show that the join happens.
        words = "hallo welt" if task_sym == "<st_deu>" else "hello world"
        first, second = words.split()
        return [(0.0, 1.0, first), (1.0, 2.0, second)]


def test_transcribe_maps_codes_to_symbols(monkeypatch, wav):
    fake = FakeASR()
    monkeypatch.setattr(mcp_server, "_asr", lambda: fake)
    server = mcp_server.build_server()
    assert call(server, "transcribe", audio_path=str(wav)) == "hello world"
    assert (
        call(server, "translate", audio_path=str(wav), language="eng", to="deu")
        == "hallo welt"
    )
    assert fake.calls == [
        (str(wav), "<nolang>", "<asr>"),
        (str(wav), "<eng>", "<st_deu>"),
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
    with pytest.raises(ToolError, match="target"):
        asyncio.run(
            server.call_tool("translate", {"audio_path": str(wav), "to": "fra"})
        )
    with pytest.raises(ToolError, match="No such audio file"):
        asyncio.run(
            server.call_tool("transcribe", {"audio_path": str(tmp_path / "no.wav")})
        )


class FakePOWSM:
    """POWSM writes each phone between slashes; the server takes them out."""

    def __init__(self):
        self.calls = []
        self.s2t_model = SimpleNamespace(token_list=["<unk>", "<eng>", "<pr>"])

    def no_language(self):
        return "<unk>"

    def decode_long(self, speech, lang_sym, task_sym):
        self.calls.append((speech, lang_sym, task_sym))
        return [(0.0, 1.0, "/ð//ə/"), (1.0, 2.0, "/s//eɪ//l/")]


def test_phonemize_returns_one_token_a_phone(monkeypatch, wav):
    fake = FakePOWSM()
    monkeypatch.setattr(mcp_server, "_pr", lambda: fake)
    server = mcp_server.build_server()

    assert call(server, "phonemize", audio_path=str(wav)) == "ð ə s eɪ l"

    # the model's own symbol for "work the language out yourself", which is
    # <unk> for POWSM and <nolang> for OWSM
    assert fake.calls == [(str(wav), "<unk>", "<pr>")]


def test_phonemize_takes_a_language_the_model_has(monkeypatch, wav):
    fake = FakePOWSM()
    monkeypatch.setattr(mcp_server, "_pr", lambda: fake)
    server = mcp_server.build_server()

    call(server, "phonemize", audio_path=str(wav), language="eng")
    assert fake.calls[-1][1] == "<eng>"

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
