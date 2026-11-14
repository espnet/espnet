"""Voice activity detection with two TCNs, an online one and an offline one.

The source of a Hugging Face Space, kept in espnet at
egs2/frontia_vad_v1/diar1/demo. The recipe that trained the two checkpoints is the
directory above this one. The detection itself is espnet2.bin.diar_inference
(VoiceActivityDetector) and diar_inference_streaming (StreamingVAD), which
`espnet detect` and the MCP tool `detect` call too, so the Space, the
command line and the agent give the same segments for the same file.

Three tabs:
- Recording: one file, detected by the online model and by the offline one
  (with its largest look-ahead), both drawn on the waveform;
- Live: the microphone, detected by the online model frame by frame as the
  audio arrives;
- Many files: a batch through the offline model with
  DiarizeSpeech.diarize_batch, as for cleaning a corpus.
"""

# The ZeroGPU package patches torch as it is imported, so it has to come
# first - before torch, and before anything that imports torch.
try:  # only Hugging Face's runners have it
    import spaces  # isort: skip
except ImportError:  # running elsewhere: the decorator does nothing

    class spaces:  # noqa: N801 - stands in for the module
        @staticmethod
        def GPU(func=None, **kwargs):
            return func if func is not None else (lambda f: f)


import os  # noqa: E402

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from espnet2.bin.diar_inference import (  # noqa: E402
    VAD_RULES,
    VoiceActivityDetector,
    audio_seconds,
    frames_to_segments,
    read_audio,
)

# Each recording is cut at MAX_SECS, and so is a live session. The models are
# a few hundred kilobytes and read a minute of audio in well under a second on
# a CPU; the cap keeps a shared Space responsive, not the models.
MAX_SECS = 120
GPU_SECONDS = 120
# The offline model and the online one: diar_inference.VAD_MODELS, repeated here
# because a Space is one directory (test_demo_apps.py checks they agree). Each
# takes a packed .zip from stage 8 as well as a tag.
MODEL_TAG = os.environ.get("VAD_MODEL_TAG", "espnet/frontia_vad_v1_offline")
ONLINE_MODEL_TAG = os.environ.get(
    "VAD_ONLINE_MODEL_TAG", "espnet/frontia_vad_v1_online"
)
# ZeroGPU attaches the GPU only while a @spaces.GPU function runs, so
# torch.cuda.is_available() is False here and asking it would pin the models to
# the CPU on the very hardware bought to run them. SPACES_ZERO_GPU is the
# runtime's own marker; `spaces` being importable is not, since anyone can
# install it.
ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))
if os.environ.get("DEVICE"):
    DEVICE = os.environ["DEVICE"]
elif ZERO_GPU or torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
# How many recordings the batch tab takes at once.
BATCH_FILES = 20
EXAMPLE_WAV = (
    "https://github.com/espnet/espnet/raw/master/test_utils/ctc_align_test.wav"
)

TITLE = "Voice activity detection"
DESCRIPTION = """# Voice activity detection

Two small temporal convolutional networks mark, frame by frame (8 ms), where
someone is speaking. Both were trained with
[ESPnet](https://github.com/espnet/espnet) as one-speaker diarization on
meeting, far-field and broadcast speech, by the recipe at
[`egs2/frontia_vad_v1/diar1`](https://github.com/espnet/espnet/tree/master/egs2/frontia_vad_v1/diar1).

**Online** is a causal model: it reads no future audio, so it runs on a live
stream. **Offline** reads up to about a second ahead, and places the
boundaries better for it when the whole recording is there. A frame is speech
when its probability is above the threshold, and a median filter then removes
blips shorter than half its length; each model has its own, tuned on AMI MH
dev.
"""
ARTICLE = """Source of this Space:
[`egs2/frontia_vad_v1/diar1/demo`](https://github.com/espnet/espnet/tree/master/egs2/frontia_vad_v1/diar1/demo).
The same detection from a terminal is `espnet detect recording.wav` (offline) or
`espnet detect --live` (online), and from an agent the MCP tool `detect`.
"""

offline_vad = VoiceActivityDetector.for_mode("offline", MODEL_TAG, device=DEVICE)
online_vad = VoiceActivityDetector.for_mode("online", ONLINE_MODEL_TAG, device=DEVICE)
FS = offline_vad.fs
FRAME_SHIFT = offline_vad.frame_shift
if (online_vad.fs, online_vad.frame_shift) != (FS, FRAME_SHIFT):
    raise ValueError("the online and offline models must share a rate and frames")
# The microphone's chunks arrive outside any @spaces.GPU call, where ZeroGPU
# has no GPU attached, so the live tab runs its own copy of the online model
# on the CPU; the model is a few hundred kilobytes and keeps up there.
live_vad = (
    online_vad
    if DEVICE == "cpu"
    else VoiceActivityDetector.for_mode("online", ONLINE_MODEL_TAG, device="cpu")
)
OFFLINE_INFO = (
    f"the offline model reads {offline_vad.offline_look_ahead} frames "
    f"({offline_vad.offline_look_ahead * FRAME_SHIFT:.2f} s) ahead"
)


def _read(path):
    """One recording at the models' rate, trimmed to the cap."""
    if path is None:
        raise gr.Error("Record or upload a recording first.")
    # the header gives the length, so only the part that is used is decoded
    if audio_seconds(path) > MAX_SECS:
        gr.Warning(
            f"Only the first {MAX_SECS} s were used. Run the app yourself, or "
            "`espnet detect`, for longer audio - the models have no such limit."
        )
    return read_audio(path, FS, max_seconds=MAX_SECS)


def _table(segments):
    return [
        [round(s.start, 2), round(s.end, 2), round(s.duration, 2)] for s in segments
    ]


def _plot(speech, curves):
    """The waveform, each model's probability, and its speech segments shaded."""
    seconds = np.arange(len(speech)) / FS
    fig, axes = plt.subplots(
        len(curves) + 1, 1, figsize=(10, 1.6 + 1.4 * len(curves)), sharex=True
    )
    axes[0].plot(seconds, speech, linewidth=0.4, color="0.3")
    axes[0].set_ylabel("waveform")
    for ax, (name, probs, threshold, segments) in zip(axes[1:], curves):
        frames = np.arange(len(probs)) * FRAME_SHIFT
        ax.plot(frames, probs, linewidth=0.8)
        ax.axhline(threshold, color="0.5", linestyle="--", linewidth=0.6)
        for seg in segments:
            ax.axvspan(seg.start, seg.end, alpha=0.25)
            axes[0].axvspan(seg.start, seg.end, alpha=0.08)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(name)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    # gradio draws the figure from its data; closed, pyplot does not keep it
    plt.close(fig)
    return fig


@spaces.GPU(duration=GPU_SECONDS)
def predict(path, on_threshold, on_median, off_threshold, off_median):
    speech = _read(path)
    curves = []
    for name, vad, look_ahead, threshold, median in (
        ("online", online_vad, 0, on_threshold, on_median),
        (
            "offline",
            offline_vad,
            offline_vad.offline_look_ahead,
            off_threshold,
            off_median,
        ),
    ):
        probs = vad.probabilities(speech, FS, look_ahead)
        segments = frames_to_segments(probs, FRAME_SHIFT, threshold, int(median))
        curves.append((name, probs, threshold, segments))
    offline_segments = curves[1][3]
    speech_secs = sum(s.duration for s in offline_segments)
    summary = (
        f"{len(offline_segments)} segments, {speech_secs:.1f} s of speech in "
        f"{len(speech) / FS:.1f} s (offline)"
    )
    return _plot(speech, curves), _table(offline_segments), summary


@spaces.GPU(duration=GPU_SECONDS)
def predict_files(files, threshold, median):
    if not files:
        raise gr.Error("Upload one or more recordings first.")
    if len(files) > BATCH_FILES:
        gr.Warning(f"Only the first {BATCH_FILES} files were used.")
        files = files[:BATCH_FILES]
    paths = [getattr(f, "name", f) for f in files]
    speeches = [_read(p) for p in paths]
    # a recording shorter than one frame gets no frames, and no rows
    probs = offline_vad.probabilities_batch(speeches, offline_vad.offline_look_ahead)
    rows = []
    for path, p in zip(paths, probs):
        name = os.path.basename(path)
        for seg in frames_to_segments(p, FRAME_SHIFT, threshold, int(median)):
            rows.append([name, round(seg.start, 2), round(seg.end, 2)])
    return rows


# The live plot scrolls: it shows the last LIVE_WINDOW seconds.
LIVE_WINDOW = 10.0
# Waveform drawn as its min/max envelope over 10 ms bins, which is what a
# 1000-pixel plot can show anyway and keeps each update fast.
ENVELOPE_SECS = 0.01


class LiveSession:
    """A microphone session: the online stream and the recent audio to draw."""

    def __init__(self):
        self.stream = live_vad.stream()
        self.rate = None
        self.samples = 0  # microphone samples received
        self.recent = np.zeros(0, dtype=np.float32)  # the last LIVE_WINDOW s

    def add(self, samples, rate):
        if self.rate is None:
            self.rate = rate
        self.samples += len(samples)
        keep = int(LIVE_WINDOW * self.rate)
        self.recent = np.concatenate([self.recent, samples])[-keep:]
        # the stream resamples the microphone rate as one signal across chunks
        self.stream.accept(samples, rate)


def _live_view(session, threshold, median):
    """The waveform of the last seconds with the speech shaded, and the probability.

    Shaded is what the online model has decided is speech so far, by the
    sliders' rule; the red line is now. The median filter settles a frame
    median // 2 frames after it arrives, so the edge of a segment can still
    move by that much.
    """
    probs = session.stream.all_probabilities()
    segments = frames_to_segments(probs, FRAME_SHIFT, threshold, int(median))
    rate = session.rate or FS
    now = session.samples / rate
    start = max(0.0, now - LIVE_WINDOW)
    fig, (wave_ax, prob_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 3.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    step = max(1, int(ENVELOPE_SECS * rate))
    usable = len(session.recent) // step * step
    if usable:
        bins = session.recent[len(session.recent) - usable :].reshape(-1, step)
        t = now - usable / rate + np.arange(len(bins)) * step / rate
        wave_ax.fill_between(t, bins.min(1), bins.max(1), color="0.3", linewidth=0)
    speaking = bool(segments) and segments[-1].end >= len(probs) * FRAME_SHIFT
    for seg in segments:
        if seg.end > start:
            for ax in (wave_ax, prob_ax):
                ax.axvspan(seg.start, seg.end, color="tab:green", alpha=0.3)
    frames = np.arange(len(probs)) * FRAME_SHIFT
    shown = frames >= start
    prob_ax.plot(frames[shown], probs[shown], linewidth=0.8, color="tab:blue")
    prob_ax.axhline(threshold, color="0.5", linestyle="--", linewidth=0.6)
    for ax in (wave_ax, prob_ax):
        ax.axvline(now, color="tab:red", linewidth=1.0)
    wave_ax.set_xlim(start, start + LIVE_WINDOW)
    peak = max(0.05, float(np.abs(session.recent).max()) if usable else 0.05)
    wave_ax.set_ylim(-peak, peak)
    wave_ax.set_yticks([])
    wave_ax.set_title(
        "speaking" if speaking else "silence",
        loc="left",
        color="tab:green" if speaking else "0.4",
        fontweight="bold",
    )
    prob_ax.set_ylim(-0.02, 1.02)
    prob_ax.set_ylabel("speech")
    prob_ax.set_xlabel("seconds")
    fig.tight_layout()
    plt.close(fig)
    return session, fig, _table(segments)


def live(session, chunk, threshold, median):
    """Feed one microphone chunk to the online model; return what is decided."""
    if chunk is None:
        return session, None, []
    rate, samples = chunk
    samples = np.asarray(samples)
    if samples.dtype.kind == "i":
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    if samples.ndim > 1:
        samples = samples[:, 0]
    if session is None:
        session = LiveSession()
    # what is left of the session, in samples at the microphone's rate
    allowed = int((MAX_SECS * FS - session.stream.num_samples) * int(rate) / FS)
    if len(samples) > allowed:
        gr.Warning(f"A live session stops at {MAX_SECS} s; start a new one.")
        samples = samples[: max(allowed, 0)]
    if len(samples):
        session.add(samples.astype(np.float32), int(rate))
    return _live_view(session, threshold, median)


def finish_live(session, threshold, median):
    """When the microphone stops: run the audio the resampler still holds."""
    if session is None:
        return session, None, []
    session.stream.finish()
    return _live_view(session, threshold, median)


def _rule(mode):
    """Threshold and median sliders, starting at the mode's model's tuned rule."""
    tuned_threshold, tuned_median = VAD_RULES[mode]
    threshold = gr.Slider(
        0.05,
        0.95,
        value=tuned_threshold,
        step=0.05,
        label=f"{mode.capitalize()} threshold",
        info="speech probability above which a frame is speech (tuned on AMI MH dev)",
    )
    median = gr.Slider(
        1,
        51,
        value=tuned_median,
        step=2,
        label=f"{mode.capitalize()} median filter (frames of 8 ms)",
    )
    return threshold, median


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab("Recording"):
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Recording",
                )
                on_threshold, on_median = _rule("online")
                off_threshold, off_median = _rule("offline")
                gr.Markdown(f"Offline: {OFFLINE_INFO}.")
                button = gr.Button("Detect", variant="primary")
            with gr.Column():
                summary = gr.Textbox(label="Summary")
                table = gr.Dataframe(
                    headers=["start (s)", "end (s)", "length (s)"],
                    label="Speech segments (offline)",
                )
        plot = gr.Plot(label="Online and offline")
        rule_inputs = [on_threshold, on_median, off_threshold, off_median]
        button.click(predict, [audio, *rule_inputs], [plot, table, summary])
        gr.Examples(
            [[EXAMPLE_WAV, *VAD_RULES["online"], *VAD_RULES["offline"]]],
            inputs=[audio, *rule_inputs],
            outputs=[plot, table, summary],
            fn=predict,
            cache_examples=False,
        )
    with gr.Tab("Live"):
        gr.Markdown(
            "Speak into the microphone: the online model decides each 8 ms "
            "frame as soon as its audio arrives, and the speech it finds is "
            "shaded on the waveform as you talk."
        )
        stream_state = gr.State()
        mic = gr.Audio(sources=["microphone"], streaming=True, label="Microphone")
        live_threshold, live_median = _rule("online")
        live_plot = gr.Plot(label="The last 10 seconds")
        live_table = gr.Dataframe(
            headers=["start (s)", "end (s)", "length (s)"], label="Speech segments"
        )
        mic.stream(
            live,
            [stream_state, mic, live_threshold, live_median],
            [stream_state, live_plot, live_table],
            # a chunk every 0.2 s keeps the plot close to the voice
            stream_every=0.2,
        )
        mic.start_recording(lambda: None, None, stream_state)
        mic.stop_recording(
            finish_live,
            [stream_state, live_threshold, live_median],
            [stream_state, live_plot, live_table],
        )
    with gr.Tab("Many files"):
        gr.Markdown(
            f"Up to {BATCH_FILES} recordings, detected offline in one batch - "
            "what cleaning a corpus looks like. `espnet detect *.wav` does the "
            "same from a terminal, for any number of files."
        )
        files = gr.File(file_count="multiple", label="Recordings")
        batch_threshold, batch_median = _rule("offline")
        batch_button = gr.Button("Detect", variant="primary")
        batch_table = gr.Dataframe(
            headers=["file", "start (s)", "end (s)"], label="Speech segments"
        )
        batch_button.click(
            predict_files, [files, batch_threshold, batch_median], batch_table
        )
    gr.Markdown(ARTICLE)


if __name__ == "__main__":
    demo.launch()
