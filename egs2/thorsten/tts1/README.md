# THORSTEN-VOICE RECIPE

This is the recipe of German single male speaker TTS model with [Thorsten-Voice
Dataset 2022.10](https://zenodo.org/records/7265581) corpus, 11.2 hours of
studio-quality speech at 22.05 kHz.

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

## Recipe flow

### Data preparation

The corpus ships a train/dev/test split whose dev set holds only 4 utterances,
so `local/data.sh` merges the three metadata files and re-splits them
deterministically into 12,232 / 100 / 100. [NVIDIA NeMo's Thorsten
recipe](https://github.com/NVIDIA-NeMo/NeMo/blob/main/scripts/dataset_processing/tts/thorsten_neutral/get_data.py)
discards the released split the same way.

```sh
$ ./run.sh --stage 1 --stop-stage 1
```

### Tacotron 2 training

```sh
$ ./run.sh --stage 2 --stop-stage 7 --ngpu 4
```

### Neural vocoder

Griffin-Lim, the `tts.sh` default, is what limits this recipe rather than the
acoustic model: run on ground-truth mel-spectrograms it still scores only 1.44
UTMOS. A HiFi-GAN vocoder for this speaker is released below and raises UTMOS to
3.04 — download it and pass `--vocoder_file`.

It was fine-tuned for 40k steps from `parallel_wavegan/ljspeech_hifigan.v1`, an
exact feature match for this recipe, using
[kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
(`tools/installers/install_parallel-wavegan.sh`) on `voc1` data directories
copied from this recipe's, so it never trains on the TTS test set.

### Decoding

```sh
# Drop --vocoder_file to use Griffin-Lim.
$ ./run.sh --stage 8 --inference_model valid.loss.best.pth \
    --vocoder_file /path/to/thorsten_hifigan_ft_ljspeech/checkpoint-40000steps.pkl
```

# INITIAL RESULTS

- Initial Tacotron 2 model with a fine-tuned HiFi-GAN vocoder
- We achieve the best validation loss at epoch 29

## Environments

- date: `Wed Aug 26 00:46:57 CDT 2026`
- python version: `3.10.14`
- espnet version: `espnet 202604`
- pytorch version: `pytorch 2.8.0+cu128`
- Git hash: `2d9a6c37c8eef710debc903d86132f1ad9a40c9f`
  - Commit date: `Fri Aug 14 16:33:08 2026 -0400`

## Results

| System | MCD | log-F0 RMSE | UTMOS | WER (%) | CER (%) |
|:--|--:|--:|--:|--:|--:|
| Ground truth             |     -        | -             | 3.29 ± 0.21 |  6.2 | 3.2 |
| Tacotron 2 + Griffin-Lim | 10.27 ± 1.07 | 0.256 ± 0.056 | 1.42 ± 0.13 | 10.0 | 4.4 |
| Tacotron 2 + HiFi-GAN    |  5.83 ± 1.36 | 0.268 ± 0.061 | 3.04 ± 0.28 | 10.2 | 4.1 |

- MCD and log-F0 RMSE use DTW alignment against the ground truth. The model is
  free-running, so the duration mismatch inflates both.
- WER/CER use `openai/whisper` `medium` in German with `whisper_basic`
  normalization. The ground-truth row is non-zero mostly from orthography the
  cleaner leaves alone — numerals (`40` vs. `vierzig`) and German compounds
  (`weggegangen` vs. `weg gegangen`) — not misrecognition.
- UTMOS is trained on English, so read it only against the ground-truth row.

Evaluated with the scripts from
[Evaluation](../../TEMPLATE/tts1/README.md#evaluation):

```sh
$ gen_dir=exp/tts_train_raw_phn_espeak_ng_german/<decode_dir>/test
$ ./pyscripts/utils/evaluate_mcd.py "${gen_dir}"/wav/wav.scp dump/raw/test/wav.scp
$ ./pyscripts/utils/evaluate_f0.py "${gen_dir}"/wav/wav.scp dump/raw/test/wav.scp
$ ./pyscripts/utils/evaluate_pseudomos.py "${gen_dir}"/wav/wav.scp
$ ./scripts/utils/evaluate_asr.sh --stop_stage 3 --whisper_tag medium \
    --gpu_inference true --cleaner whisper_basic --hyp_cleaner whisper_basic \
    --decode_options "{language: de, task: transcribe, beam_size: 5}" \
    --gt_text dump/raw/test/text \
    "${gen_dir}"/wav/wav.scp exp/tts_train_raw_phn_espeak_ng_german/whisper_medium
```

## Pretrained Models

### jjiang4/thorsten_tts_train_tacotron2_raw_phn_espeak_ng_german
- https://huggingface.co/jjiang4/thorsten_tts_train_tacotron2_raw_phn_espeak_ng_german

### jjiang4/thorsten_hifigan_ft_ljspeech
- https://huggingface.co/jjiang4/thorsten_hifigan_ft_ljspeech

```python
from espnet2.bin.tts_inference import Text2Speech
from huggingface_hub import snapshot_download

vocoder = snapshot_download("jjiang4/thorsten_hifigan_ft_ljspeech")
tts = Text2Speech.from_pretrained(
    "jjiang4/thorsten_tts_train_tacotron2_raw_phn_espeak_ng_german",
    vocoder_file=f"{vocoder}/checkpoint-40000steps.pkl",
)
wav = tts("im prozess wurden aber nur vierzig fälle thematisiert.")["wav"]
```
