"""Transcribe audio as it arrives, from a file or from a microphone.

`espnet transcribe file.wav` waits for the whole file and prints once. That is the
wrong shape for a demonstration and impossible for a microphone, so this
decodes a window at a time and prints each one as it is ready.

The windows are decoded independently: a word straddling a boundary can come
out twice or not at all. Reading the whole file at once is more accurate, and
it is what `espnet transcribe` without `--stream` does; this is for watching the
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
# The most unprocessed microphone audio kept in memory. A microphone produces
# audio at exactly one second per second and the decoder consumes it as fast
# as it can, so the two only stay level while decoding is faster than real
# time. When it is not - a large model, a busy CPU, no GPU - the difference
# has to go somewhere, and the only choices are memory that grows for as long
# as the program runs or audio that is dropped. Half a minute is the ceiling.
MAX_BUFFERED_SECS = 30.0


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
    """Read an audio file block by block, resampled to what the model expects.

    Reading the file and then cutting it up would make memory proportional to
    its length - two hours of 16 kHz float32 is about 460 MB, and more before
    it is resampled - which is the opposite of what `--stream` promises. So
    one block is read, mixed to one channel and resampled at a time, and the
    only thing kept between blocks is the resampler's own state.

    That state is why this uses `soxr` directly rather than
    `librosa.resample`: resampling each block as if it were a whole recording
    puts a discontinuity at every boundary. `ResampleStream` carries its
    filter across the blocks, so the result is the same signal the one-shot
    call would have produced.
    """
    try:
        import soundfile as sf
    except ImportError as e:  # pragma: no cover - soundfile is a core dependency
        raise LiveError("soundfile is not installed: pip install espnet") from e

    with sf.SoundFile(path) as audio:
        resampler = None
        if audio.samplerate != sample_rate:
            try:
                import soxr
            except ImportError as e:  # pragma: no cover - librosa requires soxr
                raise LiveError("resampling needs soxr: pip install espnet") from e

            resampler = soxr.ResampleStream(
                audio.samplerate, sample_rate, num_channels=1, dtype="float32"
            )
        # blocks are read in the file's own rate, so that each one covers the
        # same amount of time whatever rate the file was recorded at
        read_size = max(1, round(block * audio.samplerate / sample_rate))
        while True:
            data = audio.read(read_size, dtype="float32", always_2d=False)
            # a short read is the end of the file, and so is an empty one:
            # either way this is the pass that flushes the resampler, which
            # is why it is not a `break` on an empty read
            last = len(data) < read_size
            if data.ndim > 1:
                data = data.mean(axis=1)  # a live transcript wants one channel
            if resampler is not None:
                # `last` flushes the filter's tail, so the final block is not
                # a few samples short of the recording
                data = resampler.resample_chunk(data, last=last)
            if len(data) > 0:
                yield np.asarray(data, dtype=np.float32).reshape(-1)
            if last:
                break


class BoundedBlocks:
    """A queue of audio blocks that cannot grow past a fixed ceiling.

    An unbounded queue between a microphone and a decoder turns "the decoder
    is slower than real time" into memory that grows for as long as the
    program runs, and a transcript that falls a little further behind every
    second. Neither is something the user can see happening until the
    machine is out of memory.

    So there is a ceiling, and the oldest block is what goes when it is
    reached: a live transcript is worth having because it is live, and a gap
    in it beats a transcript that is minutes behind. Every dropped block is
    counted and reported, because quietly losing speech is not something a
    transcript may do.
    """

    def __init__(
        self,
        maxsize: int,
        secs_per_block: float,
        warn_every_secs: float = 10.0,
        report: Callable[[str], None] = None,
    ):
        self._blocks: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max(1, maxsize))
        self._secs_per_block = secs_per_block
        self._warn_every = max(1, round(warn_every_secs / max(secs_per_block, 1e-9)))
        self._report = report or (lambda line: print(line, file=sys.stderr))
        self.dropped = 0

    def put(self, block: np.ndarray) -> None:
        """Add a block, dropping the oldest one if there is no room.

        Called from the audio callback, so it never blocks and never raises:
        a callback that waits for the consumer stalls the device itself.
        """
        try:
            self._blocks.put_nowait(block)
            return
        except queue.Full:
            pass
        try:
            self._blocks.get_nowait()
        except queue.Empty:  # pragma: no cover - the consumer got there first
            pass
        self.dropped += 1
        if self.dropped == 1 or self.dropped % self._warn_every == 0:
            self._report(
                "decoding is slower than the audio arrives: dropped "
                f"{self.dropped * self._secs_per_block:.0f} s of audio so far"
            )
        try:
            self._blocks.put_nowait(block)
        except queue.Full:  # pragma: no cover - room was just made
            pass

    def get(self) -> np.ndarray:
        return self._blocks.get()

    def dropped_secs(self) -> float:
        return self.dropped * self._secs_per_block


def from_microphone(sample_rate: int = SAMPLE_RATE, block: int = 4000):
    """Yield blocks from the default input device until interrupted.

    sounddevice is not a dependency of espnet: it needs PortAudio, which is a
    system package, and nothing else here records audio.

    What is held between the device and the decoder is bounded: see
    `BoundedBlocks` for what happens when decoding cannot keep up.
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

    secs_per_block = block / sample_rate
    blocks = BoundedBlocks(
        maxsize=round(MAX_BUFFERED_SECS / secs_per_block),
        secs_per_block=secs_per_block,
    )

    def collect(indata, frames, time, status):
        if status:  # an overflow means the device itself dropped blocks
            print(f"audio input: {status}", file=sys.stderr)
        blocks.put(indata.copy().reshape(-1))

    print("listening, press Ctrl-C to stop", file=sys.stderr)
    try:
        with sounddevice.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block,
            callback=collect,
        ):
            while True:
                yield blocks.get()
    finally:
        if blocks.dropped:
            print(
                f"{blocks.dropped_secs():.0f} s of audio was dropped: the "
                "decoder could not keep up. A smaller model or --device cuda "
                "would have kept it.",
                file=sys.stderr,
            )


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
