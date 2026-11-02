---
title: Speaker verification
emoji: 🔍
colorFrom: gray
colorTo: green
sdk: gradio
python_version: "3.12"
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Are these two recordings the same speaker?
tags:
  - espnet
  - speaker-verification
  - speaker-recognition
  - rawnet3
models:
  - espnet/voxcelebs12_rawnet3
---

# Speaker verification

The source of the Hugging Face Space `espnet/speaker-verification`, one of
the demos kept in this repository. It takes two recordings, turns each into
a RawNet3 embedding taken from the waveform itself, and reports the cosine
between them together with a verdict. The checkpoint is
[`voxcelebs12_rawnet3`](https://huggingface.co/espnet/voxcelebs12_rawnet3),
trained on VoxCeleb 1+2 by the recipe in the directory above this one, where
it reaches 0.739% equal error rate on VoxCeleb1-O.

It is the best of that recipe's models that a `pip install` can load. The
three that score better - `ecapa_frozen` at 0.638%, and the two that
fine-tune WavLM - reach their front end through S3PRL, whose released
version calls `torchaudio.set_audio_backend`, removed in torchaudio 2.1. A
Space installs from `requirements.txt` and nothing else, so those checkpoints
would leave this one failing to start.

The threshold is not read from the checkpoint and could not be: the recipe
records each model's equal error *rate*, not the score at which it occurs.
The slider starts at 0.36 as a place to move from, and `SPK_THRESHOLD` sets
that starting value for anyone who has measured one.

The sample rate and the length the model was trained to score both come from
the checkpoint's own preprocessor config: 16 kHz and 3 s here. A recording
shorter than that is scored anyway, with a warning that the number will be
noisy.

## Where the threshold comes from

The default is 0.36, and it is derived rather than chosen. `../README.md`
reports, for this checkpoint on VoxCeleb1-O, target trials scoring
-0.8091 ± 0.1398 at an equal error rate of 0.978%. The recipe's score is
minus the Euclidean distance between L2-normalised embeddings, so a score
`s` is a cosine of `1 - s² / 2`. Reading the target scores as normal, the
equal error point sits 2.33 standard deviations below their mean — a
distance of 1.135, which is a cosine of 0.36.

That is one operating point, measured on one corpus of read English speech,
so the app makes it a slider rather than a constant. Two things worth saying
out loud in a demo: the score only means something for speech, since two
recordings of anything else land wherever the model happens to put them; and
a recording of a voice scores like the voice, so this is not authentication.

Locally, on CPU, two chunks of the same speaker (the two halves of
`test_utils/ctc_align_test.wav`) score 0.683, and the two different speakers
in `test_utils/ctc_align_test.wav` and `test_utils/st_test.wav` score -0.063
— which is the pair the app offers as its example.

The hosted demo reads up to 30 seconds of each recording and asks ZeroGPU for
a matching slice of GPU time; longer audio is trimmed, with a warning saying
so. The model itself has no such limit, so run the app yourself for longer
recordings.

The card pins `python_version: "3.12"`. espnet requires 3.12 or 3.13, and a
Space image with an older interpreter installs no espnet at all - pip reports
"No matching distribution found" and the build fails.

`requirements.txt` asks for `espnet[spk]`, not plain `espnet`: RawNet3's
front-end is `asteroid_frontend`, which imports `asteroid_filterbanks`, and
that extra is where it comes from. A Space installs from `requirements.txt`
and nothing else, so leaving it out is a build that succeeds and a Space that
cannot start - which is how it was found. A checkpoint with a `melspec_torch`
front-end needs nothing beyond the core install, and the extra costs it one
small package.

## Running and publishing

```sh
pip install -r requirements.txt gradio
python app.py                     # http://127.0.0.1:7860
```

`DEVICE` decides where the models run and takes precedence over everything
else, so `DEVICE=cpu` is a local CPU test even on a machine with a GPU. With
`DEVICE` unset, the choice is CUDA when the runtime is ZeroGPU
(`SPACES_ZERO_GPU`, the marker Hugging Face sets — asking
`torch.cuda.is_available()` there answers False, because the GPU is attached
only while a `@spaces.GPU` function runs) or when torch reports a GPU, and CPU
otherwise.

To publish, from the ESPnet checkout:

```sh
hf auth login
hf upload espnet/speaker-verification egs2/voxceleb/spk1/demo . --repo-type space
```

The Space itself has to exist as ZeroGPU hardware (`zero-a10g`), which is a
setting on the Space rather than something in these files. This model is the
smallest of the five demos — it loads in about six seconds on a CPU and
embeds a clip in well under one — so it would also run on the free CPU tier.

## Citation

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
```
