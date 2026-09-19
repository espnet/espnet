"""Transcribe audio as it arrives, from a file or from a microphone.

`espnet asr file.wav` waits for the whole file and prints once. That is the
wrong shape for a demonstration and impossible for a microphone, so this
decodes a window at a time and prints each one as it is ready.

The windows are decoded independently: a word straddling a boundary can come
out twice or not at all. Reading the whole file at once is more accurate, and
it is what `espnet asr` without `--stream` does; this is for watching the
words appear.
"""

import queue
import sys
from typing import Callable, Iterator, Optional

import numpy as np

# OWSM is trained on 30-second windows. 20 seconds of new audio per window
# leaves room for the context the model expects on either side while keeping
# the wait between printed lines short enough to watch.
WINDOW_SECS = 20.0
SAMPLE_RATE = 16000


class LiveError(RuntimeError):
    """Something the user can fix, reported without a traceback."""


def windows(
    source: Iterator[np.ndarray], window: int, flush_partial: bool = True
) -> Iterator[np.ndarray]:
    """Regroup a stream of arbitrary blocks into fixed-length windows.

    A microphone hands over whatever its buffer held, and a file is read in
    whatever size was asked for; the model wants one length. The last window
    is shorter than the rest and is still decoded, so the end of a recording
    is not silently dropped.
    """
    buffer = np.zeros(0, dtype=np.float32)
    for block in source:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        buffer = np.concatenate([buffer, block])
        while len(buffer) >= window:
            yield buffer[:window]
            buffer = buffer[window:]
    if flush_partial and len(buffer) > 0:
        yield buffer


def from_file(path: str, sample_rate: int = SAMPLE_RATE, block: int = 4000):
    """Read an audio file in blocks, resampled to what the model expects."""
    try:
        import soundfile as sf
    except ImportError as e:  # pragma: no cover - soundfile is a core dependency
        raise LiveError("soundfile is not installed: pip install espnet") from e

    speech, rate = sf.read(path, dtype="float32", always_2d=False)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)  # a live transcript wants one channel
    if rate != sample_rate:
        import librosa

        speech = librosa.resample(speech, orig_sr=rate, target_sr=sample_rate)
    for start in range(0, len(speech), block):
        yield speech[start : start + block]


def from_microphone(sample_rate: int = SAMPLE_RATE, block: int = 4000):
    """Yield blocks from the default input device until interrupted.

    sounddevice is not a dependency of espnet: it needs PortAudio, which is a
    system package, and nothing else here records audio.
    """
    try:
        import sounddevice
    except (ImportError, OSError) as e:
        # OSError: the module imports but PortAudio is missing
        raise LiveError(
            "recording needs sounddevice and PortAudio: pip install sounddevice"
            " (on macOS also `brew install portaudio`, on Debian"
            " `apt install libportaudio2`)"
        ) from e

    blocks: "queue.Queue[np.ndarray]" = queue.Queue()

    def collect(indata, frames, time, status):
        if status:  # an overflow means blocks were dropped; say so once
            print(f"audio input: {status}", file=sys.stderr)
        blocks.put(indata.copy().reshape(-1))

    print("listening, press Ctrl-C to stop", file=sys.stderr)
    with sounddevice.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block,
        callback=collect,
    ):
        while True:
            yield blocks.get()


def transcribe(
    decode: Callable[[np.ndarray], str],
    source: Iterator[np.ndarray],
    sample_rate: int = SAMPLE_RATE,
    window_secs: float = WINDOW_SECS,
    on_text: Optional[Callable[[str], None]] = None,
) -> int:
    """Decode each window as it fills and hand the text to `on_text`.

    Ctrl-C ends a recording rather than raising: stopping is how a live
    transcript finishes, not an error.
    """
    on_text = on_text or print
    window = int(sample_rate * window_secs)
    try:
        for chunk in windows(source, window):
            text = decode(chunk)
            if text:
                on_text(text)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 130
    return 0
