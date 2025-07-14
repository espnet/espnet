# GLOBE-v2 Multi-Speaker English TTS RECIPE
# CONTRIBUTORS
Terry (Zhuoyan) Tao <terryt@usc.edu>  –  GLOBE‑v2 TTS recipe and download script

This is the recipe of the English multi-speaker TTS model with [Globe v2](https://globecorpus.github.io/) corpus.

Before running the recipe, please run 
cd egs2/globe/
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
export HF_TOKEN=downloads
huggingface-cli download MushanW/GLOBE_V2 \
  --repo-type dataset \
  --include "data/*.parquet" 

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



# INITIAL RESULTS

- 44.1 kHz VITS adaptation


## Pretrained Models
### hifitts_vits_multispeaker_22.05k
https://huggingface.co/jes3275/hifitts_vits_multispeaker_22.05k
