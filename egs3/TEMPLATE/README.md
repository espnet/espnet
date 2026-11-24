# ESPnet3 recipe templates

These directories provide defaults for recipes that use the corresponding system in `espnet3/systems/`. They are not standalone recipes: each corpus recipe supplies its own `run.py`, dataset implementation, and configuration overrides.

| Template | Purpose | Example recipe |
| --- | --- | --- |
| [esp2_asr](esp2_asr/README.md) | ESPnet2-based automatic speech recognition | [LibriSpeech 100h](../librispeech_100/esp2_asr) |
| [esp2_cls](esp2_cls/README.md) | ESPnet2-based audio classification | [MELD](../meld/esp2_cls) |
| [esp2_st](esp2_st/README.md) | ESPnet2-based speech translation | [MuST-C](../must_c/esp2_st) |
| [f5tts](f5tts/README.md) | Native ESPnet3 F5-TTS | [LibriTTS](../libritts/f5tts) |

Use the system README to understand the expected data, provided stages and metrics, and the typical recipe workflow. Corpus-specific choices belong in that recipe's `conf/` and `dataset/` directories.
