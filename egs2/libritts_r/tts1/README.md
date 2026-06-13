# LIBRITTS-R RECIPE

This is the recipe of the English multi-speaker TTS model with [LibriTTS-R](http://www.openslr.org/141) corpus.

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train with X-vector](../../TEMPLATE/tts1/README.md#multi-speaker-model-with-x-vector-training)
- [How to train with speaker ID](../../TEMPLATE/tts1/README.md#multi-speaker-model-with-speaker-id-embedding-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

# FIRST RESULTS

## Pretrained Models

## VITS Baseline @ 900 epochs (LibriTTS-R)

**Setup (summary)**
- Model: VITS (multi-speaker with x-vector)
- Corpus: LibriTTS-R
- Sampling rate: 24 kHz
- Config: `conf/tuning/train_xvector_vits.yaml`
- Evaluation: objective metrics on held-out split; UTMOS/DNSMOS/PLC and speaker-similarity included.

**Metrics**

| Metric          | Value  |
|-----------------|--------:|
| MCD             | 7.936   |
| F0 RMSE         | 51.51   |
| F0 Corr         | 0.269   |
| UTMOS           | 3.867   |
| DNSMOS Overall  | 3.169   |
| DNSMOS P.808    | 3.786   |
| PLCMOS          | 4.481   |
| Spk Similarity  | 0.717   |
| Eval Utterances | 4807    |

> Notes: Results are from the initial 900-epoch run and serve as a baseline. Exact replication may vary slightly depending on GPU/seed and preprocessing details.

---
## Pretrained Models

TBA
