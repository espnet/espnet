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
./run.sh
```

The audio is 22.05 kHz and the recipe trains at 16 kHz, so stage 3 needs a
resampler: `pip install 'espnet[recipe]'` if you hit `RuntimeError: resampy is
required to resample audio`.

## Data preparation

`local/prepare_data.py` reads an utterance-level TextGrid tier and emits
`wav.scp` + `segments` + `text`, so the ten-minute recordings are never cut up on
disk. Every annotated interval carrying speech becomes one utterance, exactly as
the corpus segments it — 46180 intervals averaging 1.56 s in train.

* **Tier.** `utt.ortho.` (orthographic) by default; `utt.prono.` holds the same
  speech as actually pronounced. Switch with
  `./run.sh --local_data_opts "--tier utt.prono."`.

  | tier | transcript |
  | --- | --- |
  | `utt.ortho.` | 제 이름은 … 아 열여섯 살 만 열여섯 살입니다 |
  | `utt.prono.` | 제 이르믄 … 아 열려서 쌀 만 열려서 싸림미다 |

* **Non-speech intervals are dropped**: `<SIL>`, `<NOISE>`, `<VOCNOISE>`,
  `<LAUGH>`, `<UNKNOWN>`, `<PRIVATE.INFO>`, and `<IVER>` (the interviewer
  talking). `<LAUGH-그래서>` means the speaker said 그래서 while laughing, so it
  becomes plain `그래서`. Any interval still holding an unrecognised `<...>` tag
  is dropped rather than trained on with a hole in its transcript.

* **Only train and valid are length-filtered.** `test` is built with no duration
  filter at all, so all 5899 annotated utterances are scored, down to 0.07 s:
  `local/data.sh` passes `--min_duration 0` for that part and asr.sh stage 4
  already skips `test_sets`. Train and valid keep `--min_wav_duration 1.0`.

The transcripts are pure Hangul plus spaces; no further text normalisation.

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
- date: `Wed Sep  9 03:14:55 CDT 2026`
- python version: `3.14.7 (main, Aug 25 2026, 14:02:56) [Clang 22.1.3 ]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.12.1+cu130`
- Git hash: `83b031bfab2b8ef522707ed9ad743eb966d7dd9d`
  - Commit date: `Tue Sep 8 21:36:32 2026 -0500`

## asr_train_asr_raw_ko_bpe2000
- Total number of ASR model parameters: 25.26 M
- ASR config: `conf/train_asr.yaml`
- Decoding config: `conf/decode_asr.yaml`
- Trained on 27165 utterances (17.0 h) for 40 epochs, 432 batches/epoch
- Scored on the full test set: all 5899 annotated utterances, nothing filtered out

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test|5899|21258|68.3|26.7|5.0|2.8|34.5|60.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test|5899|65637|85.2|10.5|4.4|2.7|17.6|60.7|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test|5899|39901|74.8|18.0|7.2|2.1|27.3|60.7|

Report **CER**. Korean WER is computed over whitespace-delimited eojeol, so a
single wrong character fails a whole token and the number runs far above CER.
A quarter of the test utterances are under half a second — fragments such as
근데 / 아 / 네 with almost no context — so applying the training set's 1.0 s floor
to test as well would report a visibly better CER while measuring less.

## Reference

Yun, Weonhee, Kyuchul Yoon, Sunwoo Park, Juhee Lee, Sungmoon Cho, Donghoon Kang,
Koonhyuk Byun, Hyeonzu Hahn and Jungsun Kim. 2015. "The Korean Corpus of
Spontaneous Speech." *Phonetics and Speech Sciences* 7(2), 103–109.
