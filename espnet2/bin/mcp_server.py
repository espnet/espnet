#!/usr/bin/env python3

"""Model Context Protocol server that lets agents call ESPnet as tools.

Start it with ``espnet-mcp`` (installed by ``pip install "espnet[mcp]"``) and
register it with the agent, for example in Claude Code, Claude Desktop or
Cursor::

    {"mcpServers": {"espnet": {"command": "espnet-mcp"}}}

The agent then sees three tools - ``transcribe``, ``synthesize`` and
``enhance`` - and calls them itself when a task needs speech recognition,
translation, synthesis or denoising. Audio moves as file paths on this
machine; nothing is uploaded anywhere. Each model is downloaded from the
Hugging Face Hub on first use and kept loaded for the rest of the session.

The defaults are the checkpoints the tools were written against and can be
swapped through the environment: ``ESPNET_MCP_ASR_MODEL`` (any OWSM-CTC
checkpoint), ``ESPNET_MCP_TTS_MODEL`` (any ESPnet text-to-speech model),
``ESPNET_MCP_ENH_MODEL`` (any single-channel enhancement model, with
``ESPNET_MCP_ENH_FS`` its sampling rate) and ``ESPNET_MCP_DEVICE``.
"""

import contextlib
import functools
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # pragma: no cover - the test extra installs mcp
    MCPServer = None

    class ToolError(Exception):  # noqa: N818 - stands in for mcp's when absent
        """Raised for input the tool cannot act on; mcp relays its message."""


ASR_MODEL = os.environ.get("ESPNET_MCP_ASR_MODEL", "espnet/owsm_ctc_v4_1B")
TTS_MODEL = os.environ.get("ESPNET_MCP_TTS_MODEL", "espnet/kan-bayashi_ljspeech_vits")
ENH_MODEL = os.environ.get(
    "ESPNET_MCP_ENH_MODEL", "espnet/Wangyou_Zhang_chime4_enh_train_enh_conv_tasnet_raw"
)
ENH_FS = int(os.environ.get("ESPNET_MCP_ENH_FS", "16000"))
if ENH_FS <= 0:
    raise ValueError(f"ESPNET_MCP_ENH_FS must be a positive integer, not {ENH_FS}")
DEVICE = os.environ.get("ESPNET_MCP_DEVICE", "cpu")

# Over stdio the protocol owns stdout, so anything a model prints while
# loading (attention.py reports a missing flash_attn that way) would be read
# as a broken message. Every model call runs with stdout pointed at stderr.
_quiet_stdout = functools.partial(contextlib.redirect_stdout, sys.stderr)


@functools.lru_cache(maxsize=None)
def _asr():  # pragma: no cover - downloads the checkpoint
    with _quiet_stdout():
        from espnet2.bin.s2t_inference import Speech2Text

        return Speech2Text.from_pretrained(
            ASR_MODEL, device=DEVICE, lang_sym="<nolang>", task_sym="<asr>"
        )


@functools.lru_cache(maxsize=None)
def _tts():  # pragma: no cover - downloads the checkpoint
    with _quiet_stdout():
        from espnet2.bin.tts_inference import Text2Speech

        return Text2Speech.from_pretrained(TTS_MODEL, device=DEVICE)


@functools.lru_cache(maxsize=None)
def _enh():  # pragma: no cover - downloads the checkpoint
    with _quiet_stdout():
        from espnet2.bin.enh_inference import SeparateSpeech

        return SeparateSpeech.from_pretrained(ENH_MODEL, device=DEVICE)


# mcp relays a ToolError's message to the agent word for word and reduces
# any other exception to "Error executing tool <name>", so every check an
# agent could act on raises ToolError.
def _existing_file(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_file():
        raise ToolError(f"No such audio file: {path}")
    return p


def _symbol(model, symbol: str, what: str) -> str:
    if symbol not in model.s2t_model.token_list:
        raise ToolError(
            f"{what} {symbol!r} is not in the model's vocabulary. Use an ISO 639-3 "
            "code such as eng, jpn, deu, zho, fra or spa."
        )
    return symbol


def transcribe(
    audio_path: str, language: str = "auto", translate_to: Optional[str] = None
) -> str:
    """Transcribe a speech recording, or translate it into another language.

    Args:
        audio_path: Local audio file. Any common format, sample rate or length;
            long recordings are processed in 30-second windows.
        language: ISO 639-3 code of the spoken language (eng, jpn, deu, zho,
            fra, spa, ...) or "auto" to let the model detect it.
        translate_to: ISO 639-3 code to translate the speech into (deu, jpn,
            zho, fra, spa, ...). Leave unset to transcribe in the spoken language.

    Returns:
        The transcript or translation as plain text. The first call downloads
        the model (about 4 GB) and takes minutes; later calls take 10-30 s per
        30 s of audio on a laptop CPU, about a second on a GPU.
    """
    path = _existing_file(audio_path)
    model = _asr()
    lang_sym = (
        "<nolang>"
        if language == "auto"
        else _symbol(model, f"<{language}>", "language")
    )
    task_sym = (
        _symbol(model, f"<st_{translate_to}>", "translation target")
        if translate_to
        else "<asr>"
    )
    with _quiet_stdout():
        # one recording of any length, decoded on the CTC head: the same
        # route batch_decode took before it was deprecated
        return " ".join(
            text
            for _, _, text in model.decode_long(
                str(path), lang_sym=lang_sym, task_sym=task_sym
            )
        )


def synthesize(text: str, output_path: str) -> str:
    """Synthesize speech from text and save it as a WAV file.

    Args:
        text: The sentence(s) to speak. The default model speaks English.
        output_path: Where to write the WAV file; parent directories are created.

    Returns:
        The absolute path of the WAV file written. The first call downloads
        the model (about 400 MB).
    """
    model = _tts()
    if model.fs is None:
        # Text2Speech.fs is None when neither the model nor its vocoder
        # states a rate, and soundfile needs one to write the file.
        raise ToolError(
            f"{TTS_MODEL} does not state a sampling rate, so its output cannot "
            "be written; choose another model with ESPNET_MCP_TTS_MODEL."
        )
    with _quiet_stdout():
        wav = model(text)["wav"]
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(out), wav.cpu().numpy(), model.fs)
    return str(out.resolve())


def enhance(audio_path: str, output_path: str) -> str:
    """Remove background noise from a speech recording and save the result.

    Args:
        audio_path: Local audio file with one speaker in noise. Stereo input is
            reduced to its first channel; any sample rate is accepted.
        output_path: Where to write the enhanced WAV file (16 kHz by default);
            parent directories are created.

    Returns:
        The absolute path of the WAV file written. The first call downloads
        the model (about 30 MB).
    """
    path = _existing_file(audio_path)
    speech, fs = soundfile.read(str(path), dtype="float32", always_2d=True)
    speech = speech[:, 0]
    if fs != ENH_FS:
        import librosa

        speech = librosa.resample(speech, orig_sr=fs, target_sr=ENH_FS)
    model = _enh()
    with _quiet_stdout():
        enhanced = model(speech[None, :], fs=ENH_FS)[0]
    enhanced = np.asarray(enhanced.cpu() if hasattr(enhanced, "cpu") else enhanced)
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(out), enhanced.reshape(-1), ENH_FS)
    return str(out.resolve())


TOOLS = (transcribe, synthesize, enhance)


def build_server():
    """Return an MCPServer with the three ESPnet tools registered."""
    if MCPServer is None:
        raise ImportError(
            "`mcp` is not available. Please install it via `pip install mcp` "
            'or `pip install "espnet[mcp]"`.'
        )
    server = MCPServer(
        "espnet",
        instructions=(
            "Speech tools from ESPnet. transcribe: speech recognition in 150 "
            "languages and speech translation into 25 (OWSM-CTC). synthesize: "
            "text-to-speech to a WAV file. enhance: denoise a recording. Audio is "
            "passed as file paths on this machine."
        ),
    )
    for fn in TOOLS:
        server.add_tool(fn)
    return server


def main():  # pragma: no cover - blocks on stdio; exercised by hand
    """Serve the tools over stdio; this is the `espnet-mcp` command."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
