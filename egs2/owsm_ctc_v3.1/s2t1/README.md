# OWSM-CTC v3.1

[OWSM-CTC](https://aclanthology.org/2024.acl-long.549/) is an encoder-only speech foundation model based on hierarchical multi-task self-conditioned CTC.
This version is trained on 180k hours of public audio data for multilingual speech recognition, any-to-any speech translation, and language identification, which follows the design of the project, [Open Whisper-style Speech Model (OWSM)](https://arxiv.org/abs/2401.16658).

## Data Preparation

The training data follows the same format as the encoder-decoder OWSM v3.1, except that timestamps are removed from the `text` file. Please first follow the `egs2/owsm_v3.1/s2t1` recipe to prepare OWSM data, and then convert `text` into the new format by running `python local/convert_owsm_data.py` (the path to the BPE tokenizer needs to be modified to your path).

### OWSM-CTC Data Format

The prepared data directory contains the following files:

```
dump/raw/train
├── feats_type
├── spk2utt
├── text
├── text.ctc
├── text.prev
├── utt2spk
├── wav.scp
```

`feats_type` has a single line of text, which should be automatically generated in the data preparation stage:
```
raw
```

`spk2utt` and `utt2spk` have the same meaning as the standard Kaldi recipes (see `asr1` recipe for example). Typically, the speaker information is not utilized. Hence, each utterance has a unique speaker ID which is simply its utterance ID.

`wav.scp` also follows the standard Kaldi format.

`text` contains the multitask reference (ASR or ST) with language and task tokens but without timestamps:

```
AIDATATANG_200ZH_T0055G0013S0001_000000000_000003561_zho_asr <zho><asr> 今天什么日子
...
GigaST_YOU0000009624_002208970_002218840_en_st_zh <eng><st_zho> 大会结束后,我们要求有兴趣进一步参与我们项目或进一步参与气候教育的学生站出来,
...
MLS_en_sikhreligion6_22_macauliffe_64kb_003555300_003571720_en_asr <eng><asr> it farid considered that faqiri or holiness consisted in four things namely to be blind to the faults of muhammadans to be deaf to slander to be dumb when evil speaking is suggested and to be lame when there is a desire to visit evil places
...
```

`text.ctc` contains the pure ASR reference:

```
AIDATATANG_200ZH_T0055G0013S0001_000000000_000003561_zho_asr 今天什么日子
...
CoVoST2_147d94ad8405722d5930a859295bfac7b925ccd40c587334d34f3ebd2668a70242240866e93907398f10b7f2265a4ddb82b5355eb21fe37993d04a69900df388-common_voice_en_19741894_000000000_000006270_en_st_ca He appointed military officers to most leading government positions.
...
```

`text.prev` contains the previous sentence that will be used as an additional prompt. If a sample does not have a prompt, then `<na>` is used.

```
AIDATATANG_200ZH_T0055G0013S0001_000000000_000003561_zho_asr <na>
...
GigaST_YOU0000009624_002208970_002218840_en_st_zh 与员工和同事一起，这将有助于为事物创造空间，帮助为我们创造空间，一些掩护，尝试新事物，
...
```

## Pre-trained Model

The pre-trained model is available at: https://huggingface.co/pyf98/owsm_ctc_v3.1_1B

The model page also contains example usage.
