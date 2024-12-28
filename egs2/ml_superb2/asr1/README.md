# ML-SUPERB 2.0 2024 Challenge


This is a recipe to reproduce the baseline model for the [Interspeech 2024 ML-SUPERB 2.0 Challenge](multilingual.superbbenchmark.org). While the challenge is open-ended, the organizers have provided here a minimal training and development set based off of the [ML-SUPERB 2.0 Benchmark](https://www.isca-archive.org/interspeech_2024/shi24g_interspeech.pdf) for participants to use. This data will cover most of the evaluated languages. More information about the challenge and the dataset construction can be found on the [challenge website](https://multilingual.superbbenchmark.org/challenge-interspeech2025/challenge_overview).


The baseline uses frozen SSL features from [MMS 1B](https://www.jmlr.org/papers/v25/23-1318.html), which are input into a 2-layer Transformer trained using CTC loss. It takes roughly 2 days to train on a single H100 GPU.


The challenge will use a custom scoring script, which considers worst language performance and CER standard deviation in addition to the typical multilingual ASR metrics of language identification accuracy and ASR CER. The exact implementation can be found in `local/score.py`.

## RESULTS

### train_asr.yaml (Frozen MMS 1B + Transformer + CTC)

### Environments
- date: `Sat Dec 28 11:08:07 CST 2024`
- python version: `3.10.15 (main, Oct  3 2024, 07:21:53) [GCC 11.2.0]`
- espnet version: `espnet 202409`
- pytorch version: `pytorch 2.6.0.dev20241008+cu124`
- model_link: https://huggingface.co/espnet/mms_1b_mlsuperb
- Git hash: `4fe2783ef85c294af19f36fb519ec62dc6639ce7`
  - Commit date: `Fri Dec 27 14:11:37 2024 +0000`

|decode_dir|Standard CER|Standard LID|Worst 15 CER|CER StD|Dialect CER|Dialect LID|
|---|---|---|---|---|---|---|
decode_asr_asr_model_valid.loss.ave|0.24|0.74|0.71|0.26|0.4|0.36|
