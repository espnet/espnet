# LJSpeech TTS

This recipe ports the `egs2/ljspeech/tts1` VITS setup into `egs3/ljspeech/tts`.

Expected dataset layout:

- Set `LJSPEECH` to the directory containing `LJSpeech-1.1/`
- Or pass `--training_config` with `create_dataset.corpus_root=/path/to/LJSpeech-1.1`

Run:

```bash
./run.sh
```

Notes:

- The default frontend follows the `egs2/ljspeech/tts1` recipe: phoneme tokens with `g2p_en_no_space`.
- This recipe assumes the required TTS extras for G2P are available in the environment.
