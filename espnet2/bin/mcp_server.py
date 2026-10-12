#!/usr/bin/env python3

"""Model Context Protocol server that lets agents call ESPnet as tools.

Start it with ``espnet-mcp`` (installed by ``pip install "espnet[mcp]"``) and
register it with the agent, for example in Claude Code, Claude Desktop or
Cursor::

    {"mcpServers": {"espnet": {"command": "espnet-mcp"}}}

The agent then sees six tools - ``transcribe``, ``translate``, ``phonemize``,
``align``, ``synthesize`` and ``enhance`` - and calls them itself when a task
needs one. They are named after the subcommands of ``espnet``, so a person
reading an agent's transcript and a person at a terminal are talking about
the same thing. Audio moves as file paths on this machine; nothing is
uploaded anywhere. Each model is downloaded from the Hugging Face Hub on
first use and kept loaded for the rest of the session.

The defaults are the checkpoints the tools were written against and can be
swapped through the environment: ``ESPNET_MCP_ASR_MODEL`` (any OWSM-CTC
checkpoint), ``ESPNET_MCP_TTS_MODEL`` (any ESPnet text-to-speech model),
``ESPNET_MCP_ENH_MODEL`` (any single-channel enhancement model, with
``ESPNET_MCP_ENH_FS`` its sampling rate), ``ESPNET_MCP_PR_MODEL`` (a POWSM
checkpoint, for the phones) and ``ESPNET_MCP_DEVICE`` (``auto`` by default:
cuda when torch sees one, else cpu). Alignment uses the recognition model,
since it is a Viterbi path through that model's CTC head.

The speech-to-text tools run on espnet3's inference API: the model is an
``esp2_s2t`` ``Inference``, which takes the language and task by name and
holds what there is to know about the checkpoint; this server only maps
tool arguments to it and its sentences to ``ToolError``.
"""

import contextlib
import functools
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import soundfile

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # pragma: no cover - the test extra installs mcp
    MCPServer = None

    class ToolError(Exception):  # noqa: N818 - stands in for mcp's when absent
        """Raised for input the tool cannot act on; mcp relays its message."""


PHONE = re.compile(r"/([^/]+)/")

ASR_MODEL = os.environ.get("ESPNET_MCP_ASR_MODEL", "espnet/owsm_ctc_v4_1B")
TTS_MODEL = os.environ.get("ESPNET_MCP_TTS_MODEL", "espnet/kan-bayashi_ljspeech_vits")
# The model `espnet enhance` and the universal-se Space load, so that the
# three front ends enhance a recording the same way. It is the CHiME-4
# Conv-TasNet's replacement here: that one was trained on one corpus at one
# rate, and an agent hands this tool whatever file it was given.
ENH_MODEL = os.environ.get(
    "ESPNET_MCP_ENH_MODEL",
    "espnet/Wangyou_Zhang_universal_train_enh_uses_refch0_2mem_raw",
)
# POWSM, the phonetic model built on OWSM. The CTC one: it reads a recording
# of any length, where the encoder-decoder one repeats itself on the padding.
PR_MODEL = os.environ.get("ESPNET_MCP_PR_MODEL", "espnet/powsm_ctc")
ENH_FS = int(os.environ.get("ESPNET_MCP_ENH_FS", "16000"))
if ENH_FS <= 0:
    raise ValueError(f"ESPNET_MCP_ENH_FS must be a positive integer, not {ENH_FS}")
DEVICE = os.environ.get("ESPNET_MCP_DEVICE", "auto")

# Over stdio the protocol owns stdout, so anything a model prints while
# loading (attention.py reports a missing flash_attn that way) would be read
# as a broken message. Every model call runs with stdout pointed at stderr.
_quiet_stdout = functools.partial(contextlib.redirect_stdout, sys.stderr)


def _device() -> str:
    """The device, with ``auto`` settled against this machine."""
    from espnet3.api.inference import resolve_device

    return resolve_device(DEVICE)


def _s2t(tag: str):
    """A speech-to-text checkpoint behind espnet3's ``esp2_s2t`` Inference."""
    from espnet3.api.inference import load

    with _quiet_stdout():
        return load(tag, system="s2t", device=_device())


@functools.lru_cache(maxsize=None)
def _asr():  # pragma: no cover - downloads the checkpoint
    return _s2t(ASR_MODEL)


@functools.lru_cache(maxsize=None)
def _pr():  # pragma: no cover - downloads the checkpoint
    return _s2t(PR_MODEL)


@functools.lru_cache(maxsize=None)
def _aligner():  # pragma: no cover - downloads the checkpoint
    with _quiet_stdout():
        from espnet2.bin.align import ForcedAligner

        # the recognition model again: alignment is a Viterbi path through
        # the CTC head this tool's transcribe already reads
        return ForcedAligner.from_pretrained(ASR_MODEL, device=_device())


@functools.lru_cache(maxsize=None)
def _tts():  # pragma: no cover - downloads the checkpoint
    with _quiet_stdout():
        from espnet2.bin.tts_inference import Text2Speech

        return Text2Speech.from_pretrained(TTS_MODEL, device=_device())


@functools.lru_cache(maxsize=None)
def _enh():  # pragma: no cover - downloads the checkpoint
    with _quiet_stdout():
        from espnet2.bin.enh_inference import SeparateSpeech

        return SeparateSpeech.from_pretrained(ENH_MODEL, device=_device())


# mcp relays a ToolError's message to the agent word for word and reduces
# any other exception to "Error executing tool <name>", so every check an
# agent could act on raises ToolError.
def _existing_file(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_file():
        raise ToolError(f"No such audio file: {path}")
    return p


def _say(model, audio_path: str, task: str, language: str) -> str:
    """The model's text for one recording of any length.

    ``task`` and ``language`` are passed by name; ``"auto"`` leaves the
    language to the model. A name the checkpoint lacks is the model's own
    sentence about it - which code to use, which targets it has - relayed
    as a ``ToolError`` for the agent to act on.
    """
    path = _existing_file(audio_path)
    prompt = {"task": task}
    if language != "auto":
        prompt["language"] = language
    try:
        with _quiet_stdout():
            return model(str(path), **prompt)["text"]
    except ValueError as e:
        raise ToolError(str(e)) from e


def transcribe(audio_path: str, language: str = "auto") -> str:
    """Transcribe a speech recording.

    Args:
        audio_path: Local audio file. Any common format, sample rate or length;
            a long recording is decoded in windows of the model's own length.
        language: ISO 639-3 code of the spoken language (eng, jpn, deu, zho,
            fra, spa, ...) or "auto" to let the model detect it.

    Returns:
        The transcript as plain text. The first call downloads the model -
        the default is about 4 GB - and takes minutes; later calls take
        10-30 s per 30 s of audio on a laptop CPU, about a second on a GPU.
    """
    return _say(_asr(), audio_path, "asr", language)


def _warn_if_untrained_direction(language: str, to: str) -> None:
    """Say so when neither side of the translation is English.

    OWSM's speech translation is trained on data that pairs English with the
    other language, so a direction with English on neither side is zero-shot
    - raised in review on #6780. The model has one target symbol per
    language and none for the pair, so this is the only place the direction
    is known. With `language="auto"` the source is decided inside the model,
    which is why that case is documented rather than warned about.
    """
    if to != "eng" and language not in ("auto", "eng"):
        warnings.warn(
            f"{language} to {to} translation has English on neither side, "
            "which is outside what the model was trained on; the output may "
            "be unreliable",
            UserWarning,
            stacklevel=3,
        )


def translate(audio_path: str, to: str, language: str = "auto") -> str:
    """Translate a speech recording into another language, as text.

    The training data pairs English with the other language, so translation
    into or out of English is what the model was taught. A direction with
    English on neither side - Japanese into German, say - is outside that:
    it decodes, and may be right, but nothing about it was trained.

    Args:
        audio_path: Local audio file, as for `transcribe`.
        to: ISO 639-3 code to translate into (eng, deu, jpn, zho, fra, spa,
            ...). A code the model was not trained for is reported with the
            ones it has.
        language: ISO 639-3 code of the spoken language, or "auto".

    Returns:
        The translation as plain text.
    """
    _warn_if_untrained_direction(language, to)
    return _say(_asr(), audio_path, f"st_{to}", language)


def phonemize(audio_path: str, language: str = "auto") -> str:
    """Recognise the phones in a recording, as IPA.

    What was said, in sounds rather than words: useful for pronunciation
    work, for a language with no written form to hand, and for lining speech
    up with a lexicon.

    Args:
        audio_path: Local audio file, as for `transcribe`.
        language: ISO 639-3 code of the spoken language, or "auto".

    Returns:
        The phones, space-separated, one token per phone: "ð ə s eɪ l".
    """
    decoded = _say(_pr(), audio_path, "pr", language)
    # POWSM writes each phone between slashes, so that a phone spelled like a
    # BPE token is still one token
    phones = PHONE.findall(decoded)
    return " ".join(phones) if phones else decoded


def align(audio_path: str, utterances: List[str]) -> str:
    """Find when each utterance was said in a recording.

    Args:
        audio_path: Local audio file, as for `transcribe`.
        utterances: The lines that were said, in the order they were said.
            Each is aligned to a stretch of the recording.

    Returns:
        One line an utterance: start seconds, end seconds, how sure the
        alignment is, and the text, separated by tabs. The score is the mean
        probability of the utterance's tokens, so 1.0 is a perfect match and
        a caption that does not belong to the audio scores near zero.
    """
    path = _existing_file(audio_path)
    lines = [str(u).strip() for u in utterances if str(u).strip()]
    if not lines:
        raise ToolError("Give the utterances to align, as a list of strings.")
    try:
        with _quiet_stdout():
            segments = _aligner()(str(path), lines)
    except ValueError as e:
        # "this text cannot fit in this recording", and the like
        raise ToolError(str(e)) from e
    return "\n".join(
        f"{s.start:.2f}\t{s.end:.2f}\t{s.score:.4f}\t{s.text}" for s in segments
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
        the model (about 15 MB).
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


TOOLS = (transcribe, translate, phonemize, align, synthesize, enhance)


def build_server():
    """Return an MCPServer with the ESPnet tools registered."""
    if MCPServer is None:
        raise ImportError(
            "`mcp` is not available. Please install it via `pip install mcp` "
            'or `pip install "espnet[mcp]"`.'
        )
    server = MCPServer(
        "espnet",
        instructions=(
            "Speech tools from ESPnet, named after the `espnet` subcommands. "
            "transcribe: speech recognition in 150 languages (OWSM-CTC). "
            "translate: speech into text in another language, 25 of them. "
            "phonemize: the phones that were said, as IPA (POWSM). "
            "align: when each utterance was said. "
            "synthesize: text-to-speech to a WAV file. "
            "enhance: denoise a recording. Audio is passed as file paths on "
            "this machine."
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
