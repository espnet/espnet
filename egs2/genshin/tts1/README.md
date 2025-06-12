# GENSHIN RECIPE

This is recipe for the multi-language and multi-speaker TTS model with dubbing corpus of video game Genshin Impact. The default recipe is EN, you can change it in local/data.sh.

You can also use this recipe for ACGN game dubbing dataset gathered by [AI Hobbyist](https://github.com/AI-Hobbyist).

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train with speaker ID](../../TEMPLATE/tts1/README.md#multi-speaker-model-with-speaker-id-embedding-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

# INITIAL RESULTS

Model: VITS

## Environments
- date: `Thu Dec 26 08:10:09 UTC 2024`
- python version: `3.11.10 (main, Oct  3 2024, 07:29:13) [GCC 11.2.0]`
- espnet version: `espnet 202412`
- pytorch version: `pytorch 2.5.1`
- Git hash: `0fa63ed0a4dae8ac19fd489ff1a14a9b8a98dd64`
  - Commit date: `Thu Dec 26 07:30:35 2024 +0000`

### MCD

Average: 7.8015 ± 2.1567

### F0 RMSE

Average: 0.3741 ± 0.0253

### Pseudo MOS

Average: 2.5120 ± 0.7532

### CER(with whisper medium)

| SPKR       | #Snt | #Wrd  | Corr | Sub | Del | Ins | Err | S.Err |
|------------|-------|--------|------|-----|-----|-----|-----|-------|
| Sum/Avg    | 4216  | 403902 | 87.8 | 6.5 | 5.7 | 2.7 | 14.9| 92.4  |
| Mean       | 40.5  | 3883.7 | 88.4 | 5.9 | 5.7 | 2.5 | 14.0| 91.9  |
| S.D.       | 101.9 | 7230.7 | 5.2  | 1.9 | 4.4 | 0.9 | 5.5 | 6.2   |
| Median     | 29.0  | 2870.5 | 89.7 | 5.6 | 4.5 | 2.2 | 12.5| 92.2  |

## Pretrained Models

### espnet/egs2/genshin/tts1/exp/44k/tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space

- https://huggingface.co/WhaleDolphin/Genshin-vits-espnet2
