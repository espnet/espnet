"""Speaker verification with ECAPA-TDNN trained on VoxCeleb 1+2.

Published as https://huggingface.co/spaces/espnet/speaker-verification; the
source lives in espnet, at egs2/voxceleb/spk1/demo. The recipe that trained
the checkpoint is the directory above this one.

One of the five demo Spaces this repository maintains; the others are
egs2/owsm_ctc_v4/s2t1/demo, egs2/owsm_v4/s2t1/demo, egs2/ljspeech/tts1/demo
and egs2/universal_se_v1/enh1/demo.
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
import torch  # noqa: E402

from espnet2.bin.spk_inference import Speech2Embedding  # noqa: E402

# Two recordings per call, each capped, and one forward pass over each: the
# model is small and the slice below is mostly the cold start. Longer audio
# is trimmed rather than refused.
MAX_SECS = 30
GPU_SECONDS = 60
MODEL_TAG = os.environ.get("SPK_MODEL_TAG", "espnet/voxcelebs12_rawnet3")
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

# egs2/voxceleb/spk1/README.md, this checkpoint on VoxCeleb1-O: target trials
# score -0.8091 +/- 0.1398 at an EER of 0.978%. The recipe scores a trial as
# minus the Euclidean distance between L2-normalised embeddings, so a score s
# is a cosine of 1 - s^2 / 2. Reading the target scores as normal, the equal
# error point sits 2.33 standard deviations below their mean: a distance of
# 1.135, which is a cosine of 0.36. That is where the slider starts. It is
# one operating point measured on one corpus of read English speech, not a
# constant of nature, which is why it is a slider and not a constant.
# Not something a checkpoint records, and not something the recipe records
# either: its table gives each model's equal error *rate*, not the score at
# which that rate occurs. A threshold is a choice about which errors to make,
# so 0.36 is a place to start moving the slider from rather than a
# measurement, and SPK_THRESHOLD sets that starting value for anyone who has
# measured one for their checkpoint.
THRESHOLD = float(os.environ.get("SPK_THRESHOLD", "0.36"))
FIRST_WAV = "https://github.com/espnet/espnet/raw/master/test_utils/ctc_align_test.wav"
SECOND_WAV = "https://github.com/espnet/espnet/raw/master/test_utils/st_test.wav"


THRESHOLD_INFO = (
    f"{THRESHOLD} is a place to start, not a measured operating point; "
    "set SPK_THRESHOLD if you have one"
)

TITLE = "Speaker verification"
DESCRIPTION = """# Speaker verification

RawNet3 turns a recording into one vector that describes the voice rather
than the words, reading the waveform itself instead of a spectrogram. Two
recordings of the same person land close together, two people land far
apart, and the cosine between the vectors is the score. This checkpoint was
trained on VoxCeleb 1+2 with [ESPnet](https://github.com/espnet/espnet), by
the recipe at
[`egs2/voxceleb/spk1`](https://github.com/espnet/espnet/tree/master/egs2/voxceleb/spk1),
where it reaches 0.74% equal error rate on VoxCeleb1-O.

Give it two recordings of speech. The verdict is the score against the
threshold, and the threshold is yours to move.
"""
ARTICLE = """Model:
[`espnet/voxcelebs12_rawnet3`](https://huggingface.co/espnet/voxcelebs12_rawnet3).
Source of this Space:
[`egs2/voxceleb/spk1/demo`](https://github.com/espnet/espnet/tree/master/egs2/voxceleb/spk1/demo).

The score means something only for speech: two recordings of anything else
land wherever the model happens to put them. A demo is also not an
authentication system - a recording of a voice scores like the voice.

```bibtex
@article{jung2024espnet,
  title={{ESPnet-SPK}: full pipeline speaker embedding toolkit with
         reproducible recipes, self-supervised front-ends, and off-the-shelf
         models},
  author={Jung, Jee-weon and Zhang, Wangyou and Shi, Jiatong and
          Aldeneh, Zakaria and Higuchi, Takuya and Theobald, Barry-John and
          Abdelaziz, Ahmed Hussen and Watanabe, Shinji},
  journal={Proc. Interspeech 2024},
  year={2024}
}
```"""


spk = Speech2Embedding.from_pretrained(MODEL_TAG, device=DEVICE)

# Both numbers come from the checkpoint rather than from this file: the rate
# it expects, and the length of the crops it was trained to score, which is
# what makes a much shorter recording worth a warning.
PREPROCESSOR = getattr(spk.spk_train_args, "preprocessor_conf", None) or {}
SAMPLE_RATE = int(PREPROCESSOR.get("sample_rate", 16000))
TRAINED_SECS = float(PREPROCESSOR.get("target_duration", 3.0))


def _read(path, which):
    """One recording at the model's rate, trimmed to the cap."""
    if path is None:
        raise gr.Error(f"Record or upload the {which} recording too.")
    # The header says how long the file is, so only the part that is scored
    # is decoded: an hour of audio uploaded to a 30 s demo is not resampled
    # in full first, and the warning is still about the file rather than
    # about what came back.
    if librosa.get_duration(path=path) > MAX_SECS:
        gr.Warning(
            f"Only the first {MAX_SECS} s of the {which} recording were used. "
            "Run the app yourself for longer audio - the model has no such "
            "limit."
        )
    speech, _ = librosa.load(path, sr=SAMPLE_RATE, duration=MAX_SECS)
    if len(speech) < SAMPLE_RATE * TRAINED_SECS:
        gr.Warning(
            f"The {which} recording is under the {TRAINED_SECS:.0f} s this "
            "model was trained to score, so the similarity will be a noisy "
            "estimate."
        )
    return speech


@spaces.GPU(duration=GPU_SECONDS)
def predict(first_path, second_path, threshold):
    first = spk(_read(first_path, "first"))
    second = spk(_read(second_path, "second"))
    # The embeddings are compared the way the recipe compares them, which is
    # a cosine: on L2-normalised vectors its score is a monotone function of
    # this one, so the threshold above carries over.
    similarity = float(torch.nn.functional.cosine_similarity(first, second).item())
    verdict = "Same speaker" if similarity >= threshold else "Different speakers"
    return round(similarity, 3), f"{verdict}, at a threshold of {threshold:.2f}"


EXAMPLES = [[FIRST_WAV, SECOND_WAV, THRESHOLD]]


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            first = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="First recording",
            )
            second = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Second recording",
            )
            threshold = gr.Slider(
                -1.0,
                1.0,
                value=THRESHOLD,
                step=0.01,
                label="Threshold",
                info=THRESHOLD_INFO,
            )
            button = gr.Button("Compare", variant="primary")
        with gr.Column():
            similarity = gr.Number(label="Cosine similarity")
            verdict = gr.Textbox(label="Verdict")
    button.click(predict, [first, second, threshold], [similarity, verdict])
    gr.Examples(
        EXAMPLES,
        inputs=[first, second, threshold],
        outputs=[similarity, verdict],
        fn=predict,
        cache_examples=False,
    )
    gr.Markdown(ARTICLE)


if __name__ == "__main__":
    demo.launch()
