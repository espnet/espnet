# KINYARWANDA TTS RECIPE

This is the first ESPnet2 recipe for Kinyarwanda TTS. It implements a Tacotron 2 architecture with character-based tokenization, utilizing transfer learning from an English LJSpeech pre-trained model to support low-resource Bantu language synthesis.

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train with speaker ID](../../TEMPLATE/tts1/README.md#multi-speaker-model-with-speaker-id-embedding-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

# INITIAL RESULTS

**Model:** Tacotron 2 (Transfer Learning from LJSpeech)
**Vocoder:** Parallel WaveGAN (LJSpeech pre-trained)

## Environments

* **date:** `Tue Feb 17 20:26:00 UTC 2026`
* **python version:** `3.12.0`
* **espnet version:** `espnet 202412`
* **pytorch version:** `pytorch 2.5.1`

### UTMOSv2 (Strong)

**Average:** 2.2103 (Evaluated on 400 test samples)

### CER (with whisper-large-v3-kin-track-b)

Evaluation performed using the Kaggle-winning ASR model for Kinyarwanda Track B to ensure a high-rigidity benchmark.

| #Snt | #Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 400 | 34270 | 86.7 | 6.1 | 7.1 | 5.8 | 19.0 | 98.2 |

### WER (with whisper-large-v3-kin-track-b)

| #Snt | #Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 400 | 4768 | 58.0 | 33.4 | 8.6 | 6.3 | 48.3 | 98.2 |

## Pretrained Models

### espnet/egs2/kinyarwanda_tts/tts1/exp/tts_kinyarwanda_transfer_v1

* [https://huggingface.co/Professor/kinyarwanda-tacotron2-espnet](https://huggingface.co/Professor/kinyarwanda-tacotron2-espnet)
