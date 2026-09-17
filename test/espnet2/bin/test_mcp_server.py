import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile
import torch

pytest.importorskip("mcp")
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from espnet2.bin import mcp_server  # noqa: E402


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
    assert set(tools) == {"transcribe", "synthesize", "enhance"}
    assert tools["transcribe"].input_schema["required"] == ["audio_path"]
    assert set(tools["transcribe"].input_schema["properties"]) == {
        "audio_path",
        "language",
        "translate_to",
    }
    assert tools["synthesize"].input_schema["required"] == ["text", "output_path"]
    assert "ISO 639-3" in tools["transcribe"].description


class FakeASR:
    def __init__(self):
        self.calls = []
        self.s2t_model = SimpleNamespace(
            token_list=["<nolang>", "<eng>", "<deu>", "<asr>", "<st_deu>"]
        )

    def batch_decode(self, speech, lang_sym, task_sym):
        self.calls.append((speech, lang_sym, task_sym))
        return "hallo welt" if task_sym == "<st_deu>" else "hello world"


def test_transcribe_maps_codes_to_symbols(monkeypatch, wav):
    fake = FakeASR()
    monkeypatch.setattr(mcp_server, "_asr", lambda: fake)
    server = mcp_server.build_server()
    assert call(server, "transcribe", audio_path=str(wav)) == "hello world"
    assert (
        call(
            server,
            "transcribe",
            audio_path=str(wav),
            language="eng",
            translate_to="deu",
        )
        == "hallo welt"
    )
    assert fake.calls == [
        (str(wav), "<nolang>", "<asr>"),
        (str(wav), "<eng>", "<st_deu>"),
    ]


def test_transcribe_rejects_what_the_model_cannot_do(monkeypatch, wav, tmp_path):
    monkeypatch.setattr(mcp_server, "_asr", FakeASR)
    server = mcp_server.build_server()
    with pytest.raises(ToolError, match="ISO 639-3"):
        asyncio.run(
            server.call_tool("transcribe", {"audio_path": str(wav), "language": "en"})
        )
    with pytest.raises(ToolError, match="translation target"):
        asyncio.run(
            server.call_tool(
                "transcribe", {"audio_path": str(wav), "translate_to": "fra"}
            )
        )
    with pytest.raises(ToolError, match="No such audio file"):
        asyncio.run(
            server.call_tool("transcribe", {"audio_path": str(tmp_path / "no.wav")})
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


def test_enhance_resamples_and_writes(monkeypatch, wav, tmp_path):
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
