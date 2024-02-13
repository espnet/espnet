# SPEECHCOMPOSER RECIPE

This is the recipe of [SpeechComposer](https://arxiv.org/pdf/2401.18045.pdf), a decoder-only speech language model which unifies multiple speech tasks with prompt composition.
See the following pages for the usage:



## Recipe flow

SpeechComposer follows VoxtLM recipe, it consists of 12 stages.

### 1. Data preparation

Data preparation stage.

If you want to add your own dataset, please create a new folder corresponding to the dataset `$your_own_dataset$` with data process bash scripts in `local`. And then add `data_${your_own_dataset}.sh` in `local`.


### 2. Wav dump / Embedding preparation

Wav dumping stage.
This stage reformats `wav.scp` in data directories.

### 3. Perform kmeans and get discrete tokens

You can change the kmeans cluster numbers via `--nclusters`.

### 4. Prepare data for different training tasks

Format data for different training tasks for train, valid, and test sets.
Preprare bpe training data.

### 5. BPE training datge

Train BPE using BPE training set obtained from last stage.

### 6. Data statistics collection

Statistics calculation stage.

### 7. Training stage

TTS model training stage.
You can change the training setting via `--lm_config` option.

### 8. Decoding for textlm and speechlm tasks.

Decoding stage.
8.a decodes for textlm task and calculates perplexity for textlm.
8.b decodes for speechlm task and calculates perplexity for speechlm.

### 9. Decoding for ASR task.

Decoding stage for ASR.
You may change the decoding setting via `--lm_inference_asr_config`. The results will be stored in the `${_scoredir}/result.txt`

### 9. Decoding for TTS task.

Decoding stage for TTS.
You may change the decoding setting via `--lm_inference_tts_config`. You may need an extra discrete vocoder to generate wavform from discrete tokens.

### 10. Decoding for VC task.

Decoding stage for voice conversion.
You may change the decoding setting via `--lm_inference_vc_config`. You may need an extra discrete vocoder to generate wavform from discrete tokens.

### 11. Decoding for SE task.

Decoding stage for speech enhancement.
You may change the decoding setting via `--lm_inference_se_config`. You may need an extra discrete vocoder to generate wavform from discrete tokens.

### 11-12. (Optional) Pack results for upload

Packing stage.
It packs the trained model files and uploads to [Zenodo](https://zenodo.org/) (Zenodo upload will be deprecated).
If you want to run this stage, you need to register your account in zenodo.
