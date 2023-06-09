# CSS10 RECIPE

This is the recipe of 10 langauge TTS model with [CSS10](https://github.com/Kyubyong/css10).

Before running the recipe, please download from https://github.com/Kyubyong/css10.
Then, edit 'CSS10' in `db.sh` and locate unzipped dataset as follows:

```bash
$ vim db.sh
CSS10=/path/to/CSS10

$ tree -L 2 /path/to/CSS10
├── de
│   ├── achtgesichterambiwasse
│   ├── meisterfloh
│   ├── serapionsbruederauswahl
│   └── transcript.txt
├── el
│   ├── Paramythi_horis_onoma
│   └── transcript.txt
├── es
│   ├── 19demarzo
│   ├── bailen
│   ├── batalla_arapiles
│   └── transcript.txt
├── fi
│   ├── ensimmaisetnovellit
│   ├── gulliverin_matkat_kaukaisilla_mailla
│   ├── kaleri-orja
│   ├── salmelan_heinatalkoot
│   └── transcript.txt
├── fr
│   ├── lesmis
│   ├── lupincontresholme
│   └── transcript.txt
├── hu
│   ├── egri_csillagok
│   └── transcript.txt
├── ja
│   ├── meian
│   └── transcript.txt
├── nl
│   ├── 20000_mijlen
│   └── transcript.txt
├── ru
│   ├── early_short_stories
│   ├── icemarch
│   ├── shortstories_childrenadults
│   └── transcript.txt
└── zh
    ├── call_to_arms
    ├── chao_hua_si_she
    └── transcript.txt
```

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train with X-vector](../../TEMPLATE/tts1/README.md#multi-speaker-model-with-x-vector-training)
- [How to train with speaker ID](../../TEMPLATE/tts1/README.md#multi-speaker-model-with-speaker-id-embedding-training)
- [How to train with language ID](../../TEMPLATE/tts1/README.md#multi-language-model-with-language-id-embedding-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)
