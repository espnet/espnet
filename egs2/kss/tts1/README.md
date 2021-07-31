# KSS RECIPE

This is the recipe of Korean female single speaker TTS model with [KSS dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset).

Before running the recipe, please download from https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset.  
Then, edit 'KSS' in `db.sh` and locate unzipped dataset as follows:

```bash
$ vim db.sh
KSS=/path/to/kss

$ tree -L 1 /path/to/kss
/path/to/kss
├── 1
├── 2
├── 3
├── 4
└── transcript.v.1.4.txt
```

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

