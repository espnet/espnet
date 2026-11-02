"""Forced alignment: when each line was said.

Published as https://huggingface.co/spaces/espnet/forced-alignment; the source
lives in espnet, at egs2/owsm_ctc_v4/s2t1/demo_align.

The other demos are one input and one output - audio in, text out - and the
page for them is built from the checkpoint by espnet2.bin.demo. This one takes
audio *and* the text that goes with it, and answers with a table, so it has a
page of its own.

The model is only a CTC head here: espnet2.bin.align aligns on whatever
checkpoint it is handed, and this Space happens to hand it OWSM-CTC v4.
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
import librosa  # noqa: E402
import matplotlib  # noqa: E402
import torch  # noqa: E402

from espnet2.bin.align import ForcedAligner  # noqa: E402

matplotlib.use("Agg")  # a Space has no display, and gradio wants the figure
import matplotlib.pyplot as plt  # noqa: E402

# ZeroGPU gives a decorated call a fixed slice of GPU time and kills it at the
# end, so the demo asks for a slice and refuses audio it could not finish in
# one. The same two minutes the other demos take.
MAX_SECS = 120
GPU_SECONDS = 120
MODEL_TAG = os.environ.get("ALIGN_MODEL_TAG", "espnet/owsm_ctc_v4_1B")
# A score is a probability under the checkpoint that produced it, so this is
# a heuristic rather than a calibrated confidence: 0.3 is where a line that
# was not said sits, on OWSM-CTC v4 and read English. ALIGN_WARN_BELOW moves
# it for a checkpoint that scores differently.
WARN_BELOW = float(os.environ.get("ALIGN_WARN_BELOW", "0.3"))
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

EXAMPLE_WAV = (
    "https://github.com/espnet/espnet/raw/master/test_utils/ctc_align_test.wav"
)
EXAMPLE_TEXT = """The sale of the hotels
is part of Holiday's strategy
to sell off assets
and concentrate on property management"""

TITLE = "Forced alignment"
DESCRIPTION = """# Forced alignment

You have a recording and the text of what was said. This says *when*: a start
and an end for every line, and a score for how well the line and the audio
agree. Subtitles come from the times; data cleaning comes from the score.

Nothing is trained here. The alignment is a Viterbi path through the CTC head
of a model that already exists - [OWSM-CTC
v4](https://huggingface.co/espnet/owsm_ctc_v4_1B) - which is why it needs no
model of its own and works in any language that one covers.

**Write the text the way the model writes it.** The score is a probability
under this model: the same words in capitals, which a reference transcript
often uses, score near zero while still landing in roughly the right place.
The app says so when it sees it.
"""
ARTICLE = """One line an utterance, in the order they were said - the times are
what you are asking for, so they are not needed. The score is the mean
probability of the line's tokens: 1.0 is a perfect match, and a line that does
not belong to this audio scores near zero, which is what alignment-score
filtering uses.

`espnet align audio.wav --text "..."` is the same thing from a terminal, and
`espnet2.bin.align.ForcedAligner` is the class behind both. Source of this
Space: [`egs2/owsm_ctc_v4/s2t1/demo_align`](https://github.com/espnet/espnet/tree/master/egs2/owsm_ctc_v4/s2t1/demo_align).
"""  # noqa: E501 - one markdown link, and breaking a URL breaks the link

aligner = ForcedAligner.from_pretrained(MODEL_TAG, device=DEVICE)
# the rate the checkpoint wants, not this file's idea of it: ALIGN_MODEL_TAG
# can point at a model trained at another rate
SAMPLE_RATE = aligner.sample_rate


def _read(path):
    if path is None:
        raise gr.Error("Record or upload some audio first.")
    speech, _ = librosa.load(path, sr=SAMPLE_RATE)
    seconds = len(speech) / SAMPLE_RATE
    if seconds > MAX_SECS:
        raise gr.Error(
            f"This demo takes up to {MAX_SECS} s; that file is {seconds:.0f} s. "
            "Run the app yourself for longer audio - the model has no such limit."
        )
    return speech


def _figure(speech, segments):
    """The waveform with each segment marked, which is the answer to look at."""
    figure, axes = plt.subplots(figsize=(11, 2.6))
    seconds = [i / SAMPLE_RATE for i in range(len(speech))]
    axes.plot(seconds, speech, linewidth=0.4, color="#888")
    for index, segment in enumerate(segments):
        axes.axvspan(segment.start, segment.end, color=f"C{index % 10}", alpha=0.25)
        axes.text(
            (segment.start + segment.end) / 2,
            0.85 * max(abs(speech.max()), 1e-6),
            segment.text.split()[0] if segment.text.split() else "",
            ha="center",
            fontsize=8,
        )
    axes.set_xlabel("seconds")
    axes.set_yticks([])
    figure.tight_layout()
    return figure


@spaces.GPU(duration=GPU_SECONDS)
def predict(audio_path, text):
    speech = _read(audio_path)
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        raise gr.Error("Type the lines that were said, one to a line.")

    try:
        segments = aligner(speech, lines)
    except ValueError as error:
        # "this text cannot fit in this recording", and the like
        raise gr.Error(str(error)) from error

    rows = [
        [f"{s.start:.2f}", f"{s.end:.2f}", f"{s.score:.3f}", s.text] for s in segments
    ]
    worst = min(s.score for s in segments)
    if worst < WARN_BELOW:
        gr.Warning(
            f"The weakest line scores {worst:.2f}, under {WARN_BELOW:.2f}. "
            "Either it was not said, or the text is spelled a way this model "
            "does not use."
        )
    return rows, _figure(speech, segments)


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(
                sources=["microphone", "upload"], type="filepath", label="Speech"
            )
            text = gr.Textbox(
                label="What was said", lines=6, placeholder="One utterance a line"
            )
            button = gr.Button("Align", variant="primary")
        with gr.Column():
            table = gr.Dataframe(
                headers=["start", "end", "score", "text"],
                label="Segments",
                wrap=True,
            )
            drawing = gr.Plot(label="Where they fall")
    button.click(predict, [audio, text], [table, drawing])
    gr.Examples(
        [[EXAMPLE_WAV, EXAMPLE_TEXT]],
        inputs=[audio, text],
        outputs=[table, drawing],
        fn=predict,
        cache_examples=False,
    )
    gr.Markdown(ARTICLE)


if __name__ == "__main__":
    demo.launch()
