# J-KAC RECIPE

This is the recipe of Japanese female single speaker TTS model with [J-KAC dataset](https://sites.google.com/site/shinnosuketakamichi/research-topics/j-kac_corpus).

Before running the recipe, please download from https://sites.google.com/site/shinnosuketakamichi/research-topics/j-kac_corpus.
Then, edit 'JKAC' in `db.sh` and locate unzipped dataset as follows:

```bash
$ vim db.sh
JKAC=/path/to/J-KAC

$ tree -L 1 /path/to/J-KAC
/path/to/J-KAC
├── pdf
├── readme.md
├── readme.pdf
├── txt
└── wav

3 directories, 2 files
```

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)
