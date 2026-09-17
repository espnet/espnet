#!/usr/bin/env python3
"""Gradio demo for OWSM-CTC: multilingual ASR, speech translation and LID.

Runs as `python app.py` and, unchanged, as a Hugging Face Space; README.md in
this directory has the upload command.
"""

import os

import gradio as gr

from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

MODEL_TAG = os.environ.get("OWSM_MODEL_TAG", "espnet/owsm_ctc_v4_1B")
DEVICE = os.environ.get("DEVICE", "cpu")
EXAMPLE_WAV = (
    "https://github.com/espnet/espnet/raw/master/test_utils/ctc_align_test.wav"
)

s2t = Speech2TextGreedySearch.from_pretrained(
    MODEL_TAG, device=DEVICE, lang_sym="<nolang>", task_sym="<asr>"
)

# OWSM token lists put the language symbols between <nolang> and <asr>, and
# every translation target is <st_xxx>. Both menus are read off the loaded
# model so they follow whichever checkpoint OWSM_MODEL_TAG names.
tokens = list(s2t.s2t_model.token_list)
LANGUAGES = [
    t[1:-1] for t in tokens[tokens.index("<nolang>") + 1 : tokens.index("<asr>")]
]
TARGETS = [t[len("<st_") : -1] for t in tokens if t.startswith("<st_")]

AUTO = "auto (detect)"
TRANSCRIBE = "transcribe"
TASKS = [TRANSCRIBE] + [f"translate to {t}" for t in TARGETS]


def run(audio, language, task):
    if audio is None:
        raise gr.Error("Record or upload some audio first.")
    lang_sym = "<nolang>" if language == AUTO else f"<{language}>"
    task_sym = "<asr>" if task == TRANSCRIBE else f"<st_{task.split()[-1]}>"
    # batch_decode reads the file itself, resamples to 16 kHz, and splits
    # anything longer than the model's 30 s window into overlapping chunks.
    return s2t.batch_decode(audio, lang_sym=lang_sym, task_sym=task_sym)


with gr.Blocks(title="OWSM-CTC v4") as demo:
    gr.Markdown(
        "# OWSM-CTC v4\n"
        f"Encoder-only speech foundation model `{MODEL_TAG}`: transcribes "
        f"{len(LANGUAGES)} languages, translates into {len(TARGETS)}, and detects "
        "the language when it is left on auto. No beam search — one encoder pass "
        f"per 30 s window, running on {DEVICE}."
    )
    with gr.Row():
        audio = gr.Audio(
            sources=["microphone", "upload"], type="filepath", label="Speech"
        )
        with gr.Column():
            language = gr.Dropdown(
                [AUTO] + LANGUAGES, value=AUTO, label="Spoken language (ISO 639-3)"
            )
            task = gr.Dropdown(TASKS, value=TRANSCRIBE, label="Task")
            button = gr.Button("Run", variant="primary")
    text = gr.Textbox(label="Output", lines=4)
    button.click(run, [audio, language, task], text)
    gr.Examples(
        [[EXAMPLE_WAV, AUTO, TRANSCRIBE]],
        inputs=[audio, language, task],
        outputs=text,
        fn=run,
        cache_examples=False,
    )
    gr.Markdown(
        "Source: [`egs2/owsm_ctc_v4/s2t1/demo`]"
        "(https://github.com/espnet/espnet/tree/master/egs2/owsm_ctc_v4/s2t1/demo) "
        "in ESPnet. Model weights: CC-BY-4.0."
    )

if __name__ == "__main__":
    demo.launch()
