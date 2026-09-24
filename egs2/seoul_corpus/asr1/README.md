# Seoul Corpus (Korean Corpus of Spontaneous Speech)

One hour of interview speech from each of 40 native Seoul-Korean speakers — 10 per
age group (teens / twenties / thirties / forties), balanced 5 male + 5 female.
Each hour ships as six ten-minute `.flac` files (240 in total, 22.05 kHz mono)
with a matching Praat `.TextGrid` label file.

Point `SEOUL_CORPUS` in `db.sh` at the directory holding the corpus; either layout
works, so nothing has to be unpacked by hand.

```
# as distributed                      # or already unpacked
$SEOUL_CORPUS/sound.tgz               $SEOUL_CORPUS/sound/*.flac
$SEOUL_CORPUS/label.tgz               $SEOUL_CORPUS/label/*.TextGrid
```

```bash
./run.sh                                                        # XEUS, the default
./run.sh --asr_config conf/tuning/train_asr_wavlm.yaml --asr_tag wavlm
```

The audio is 22.05 kHz and the recipe trains at 16 kHz, so stage 3 needs a
resampler: `pip install 'espnet[recipe]'` if you hit `RuntimeError: resampy is
required to resample audio`.

**Both configs put a frozen SSL model in front of the encoder, and that is not
optional here.** The recipe ships no from-scratch config, because training one
on this data does not converge. Both configs below clear that
floor within one epoch.

| config | upstream | needs |
| --- | --- | --- |
| `conf/tuning/train_asr_xeus.yaml` (default) | [XEUS](https://huggingface.co/espnet/xeus), multilingual | nothing; `espnet_model_zoo` fetches and caches it (2.3 GB) |
| `conf/tuning/train_asr_wavlm.yaml` | WavLM-large, English only | s3prl (`cd tools && make s3prl.done`) |

## Data preparation

`local/prepare_data.py` reads an utterance-level TextGrid tier and emits
`wav.scp` + `segments` + `text`, so the ten-minute recordings are never cut up on
disk.

* **Tier.** `utt.ortho.` (orthographic) by default; `utt.prono.` holds the same
  speech as actually pronounced. Switch with
  `./run.sh --local_data_opts "--tier utt.prono."`.

  | tier | transcript |
  | --- | --- |
  | `utt.ortho.` | 제 이름은 … 아 열여섯 살 만 열여섯 살입니다 |
  | `utt.prono.` | 제 이르믄 … 아 열려서 쌀 만 열려서 싸림미다 |

* **Non-speech events are transcription targets, not noise to discard.**
  `<SIL>`, `<NOISE>`, `<VOCNOISE>`, `<LAUGH>`, `<UNKNOWN>` and `<PRIVATE.INFO>`
  each survive as one token; `local/data.sh` writes the list to `data/nlsyms.txt`
  and `run.sh` hands it to `--bpe_nlsyms`, which keeps them atomic in the BPE
  vocabulary. `<LAUGH-그래서>` means 그래서 was said while laughing and becomes
  `<LAUGH> 그래서`. An interval holding any other `<...>` tag ends the utterance
  rather than leaving a hole in the transcript.

* **Utterances run from one interviewer turn to the next.** `<IVER>` is the
  interviewer talking, so it is dropped and is what separates utterances. Those
  spans reach 376 s, so a span longer than `--max_duration` (30 s) is cut rather
  than discarded: the cut goes on a `<SIL>` wherever the window holds one, else
  on the last interval boundary that fits. Cuts always land on an annotated
  boundary, never inside a word.

* **Each utterance is trimmed of the `<SIL>` at its two ends**, with the segment
  boundaries moved to match, so the transcript still describes exactly the audio
  in the segment. Internal silences stay. Without this, every piece produced by a
  cut ended on a silence — 24% of train and 42% of dev — which tells the model
  how the segmenter worked rather than anything about the audio. Trimming only at
  cut points would instead make a given silence transcribed or not depending on
  `--max_duration`; trimming both ends of every utterance is uniform. It costs
  17 minutes of the 26.6 h, all of it edge silence.

* **No space is written in front of a tag**, `"<SIL> 그래서<VOCNOISE>"`. The space
  after one is kept. sentencepiece splits the text at every `--bpe_nlsyms` symbol,
  so a space in front of a tag survived as a bare `▁` piece of its own — 46623 of
  them, putting `▁` at 21.7% of every target token. Dropping just that space
  removes 97% of them. Keeping the space after a tag is what makes this better
  than dropping both: a word following a tag keeps its word-initial form
  (`▁서울` rather than `서 울`).

* **CER and WER are scored without the tags**, TER with them. `run.sh` passes
  `--nlsyms_txt data/nlsyms.txt` and asr.sh removes those symbols from reference
  and hypothesis at the char and word levels. This matters: the tags are 13.7% of
  the dev tokens but 40.4% of its characters (`<VOCNOISE>` alone is 24.2%), so
  scoring them as text would measure tag spelling more than transcription.

* **Only train and dev are length-filtered** (`--min_wav_duration 1.0`,
  `--max_wav_duration 30`). `local/data.sh` passes `--min_duration 0` for `test`
  and asr.sh stage 4 already skips `test_sets`, so every annotated test utterance
  is scored.

| set | utterances | hours |
| --- | --- | --- |
| train | 7238 | 26.30 |
| dev | 810 | 3.26 |
| test | 1409 | 2.90 |

The transcripts are Hangul, spaces and the six tags; no further text normalisation.

## Splits

The 40 speakers form 8 balanced groups of 5. Each group gives up exactly one
speaker, to dev or to test, so the sets are speaker-disjoint and dev and test each
cover all four age groups with two male and two female speakers.

| set | speakers | |
| --- | --- | --- |
| train | 32 | s01–s04, s06–s09, s11–s14, s16–s19, s21–s24, s26–s29, s31–s34, s36–s39 |
| dev | 4 | s10 (f/teens), s15 (m/20s), s30 (f/30s), s35 (m/40s) |
| test | 4 | s05 (m/teens), s20 (f/20s), s25 (m/30s), s40 (f/40s) |

# RESULTS

## Environments
- date: `Sun Sep 20 23:37:00 CDT 2026`
- python version: `3.14.7 (main, Aug 25 2026, 14:02:56) [Clang 22.1.3 ]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.12.1+cu130`
- Git hash: `17f0af5119c580ee08fdd02bf324045f61be3bbe`
  - Commit date: `Thu Sep 17 15:37:48 2026 -0500`

## asr_xeus
- ASR config: `conf/tuning/train_asr_xeus.yaml`
- Decoding config: `conf/decode_asr.yaml`
- Total number of ASR model parameters: 621.43 M (47.30 M trainable, 7.6%)
- Trained on 7238 utterances (26.30 h) for 40 epochs, 1065 batches/epoch

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_num_workers0_asr_model_valid.acc.ave/test|1409|21298|69.3|26.8|3.9|3.3|34.0|73.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_num_workers0_asr_model_valid.acc.ave/test|1409|70352|90.5|6.4|3.1|2.9|12.4|69.9|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_num_workers0_asr_model_valid.acc.ave/test|1409|47522|81.9|12.3|5.8|3.4|21.5|74.6|

## asr_wavlm
- ASR config: `conf/tuning/train_asr_wavlm.yaml`
- Decoding config: `conf/decode_asr.yaml`
- Total number of ASR model parameters: 362.76 M (47.30 M trainable, 13.0%)
- Trained on 7238 utterances (26.30 h) for 40 epochs, 1065 batches/epoch

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_num_workers0_asr_model_valid.acc.ave/test|1409|21298|66.6|28.8|4.5|3.4|36.7|74.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_num_workers0_asr_model_valid.acc.ave/test|1409|70352|89.4|7.2|3.3|2.8|13.4|71.7|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_num_workers0_asr_model_valid.acc.ave/test|1409|47522|80.7|13.5|5.9|4.0|23.4|75.8|

Report **CER**. Korean WER is computed over whitespace-delimited eojeol, so a
single wrong character fails a whole token and the number runs far above CER.

XEUS is multilingual and covers Korean; WavLM-large is trained on English only
(Libri-Light, VoxPopuli, GigaSpeech), which is the likely reason for the gap.

## Reference

Yun, Weonhee, Kyuchul Yoon, Sunwoo Park, Juhee Lee, Sungmoon Cho, Donghoon Kang,
Koonhyuk Byun, Hyeonzu Hahn and Jungsun Kim. 2015. "The Korean Corpus of
Spontaneous Speech." *Phonetics and Speech Sciences* 7(2), 103–109.
