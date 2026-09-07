# Seoul Corpus (한국어 자유 발화) ASR recipe

The **Seoul Corpus** (*Korean Corpus of Spontaneous Speech*) is one hour of
interview speech from each of 40 native Seoul-Korean speakers — 10 speakers per
age group (teens / twenties / thirties / forties), balanced 5 male + 5 female.
Every hour is split into six ten-minute `.flac` files (240 in total, 22.05 kHz
mono) with a matching Praat `.TextGrid` label file.

## Getting the data ready

Point `SEOUL_CORPUS` in `db.sh` at the directory that holds the corpus. Either
layout works, so you do not have to unpack anything by hand:

```
# as distributed                      # or already unpacked
$SEOUL_CORPUS/sound.tgz               $SEOUL_CORPUS/sound/*.flac
$SEOUL_CORPUS/label.tgz               $SEOUL_CORPUS/label/*.TextGrid
```

`local/data.sh` unpacks the archives (they wrap a `.zip`) into `downloads/` when
needed, and uses an existing `sound/` + `label/` pair in place otherwise.

```bash
SEOUL_CORPUS=/path/to/seoul_corpus   # edit db.sh
./run.sh
```

The audio is 22.05 kHz and the recipe trains at 16 kHz, so stage 3 needs a
resampler: `pip install 'espnet[recipe]'` (or just `pip install resampy`) if you
hit `RuntimeError: resampy is required to resample audio`.

## What the data preparation does

`local/prepare_data.py` reads the utterance-level tier of each TextGrid and
turns every interval carrying speech into one ASR utterance
(`wav.scp` + `segments` + `text`, so the ten-minute flac files are never cut up
on disk).

* **Tier.** `utt.ortho.` (orthographic, standard spelling) by default;
  `utt.prono.` holds the same speech transcribed as actually pronounced. Switch
  with `./run.sh --local_data_opts "--tier utt.prono."`.

  | tier | transcript |
  | --- | --- |
  | `utt.ortho.` | 제 이름은 … 아 열여섯 살 만 열여섯 살입니다 |
  | `utt.prono.` | 제 이르믄 … 아 열려서 쌀 만 열려서 싸림미다 |

* **Non-speech intervals are dropped**: `<SIL>`, `<NOISE>`, `<VOCNOISE>`,
  `<LAUGH>`, `<UNKNOWN>`, `<PRIVATE.INFO>`, and `<IVER>` (the interviewer
  talking). `<LAUGH-그래서>` means the speaker said 그래서 while laughing, so it
  becomes plain `그래서`. Any interval left holding an unrecognised `<...>` tag
  is dropped rather than trained on with a hole in its transcript.

* **Short fragments are glued back together.** The utterance tier breaks at
  every pause, which alone yields ~58k fragments averaging 1.5 s. Neighbours
  separated by nothing but silence/noise are merged (`--merge_gap`, default
  0.5 s), never across the interviewer or across a dropped interval. Result:
  ~33k utterances, ~26 h, averaging 2.9 s.

The resulting transcripts are pure Hangul plus spaces — no extra text
normalisation is applied.

## Splits

The 40 speakers form 8 balanced groups of 5. One speaker per group goes to
`dev` and one to `test`, so all three sets are speaker-disjoint and stay
age- and gender-balanced.

| set | speakers | |
| --- | --- | --- |
| train | 24 | s01–s03, s06–s08, s11–s13, s16–s18, s21–s23, s26–s28, s31–s33, s36–s38 |
| dev | 8 | s04, s09, s14, s19, s24, s29, s34, s39 |
| test | 8 | s05, s10, s15, s20, s25, s30, s35, s40 |

## Baseline

`conf/train_asr.yaml` — E-Branchformer encoder (8 blocks) + Transformer decoder,
hybrid CTC/attention with `ctc_weight` 0.5, SpecAugment plus on-the-fly speed
perturbation, BPE 2000, audio resampled to 16 kHz. An external LM is configured
but off; enable it with `./run.sh --use_lm true`.

Measured on `test` (3073 utterances, 68276 characters), decoding the average of
the 10 best checkpoints:

| | Corr | Sub | Del | Ins | **Err** |
| --- | --- | --- | --- | --- | --- |
| **CER** | 86.1 | 9.9 | 4.0 | 2.2 | **16.1** |
| WER | 68.4 | 26.8 | 4.8 | 3.2 | 34.8 |
| TER | 75.3 | 18.1 | 6.6 | 2.5 | 27.1 |

Report **CER**: Korean WER is computed over whitespace-delimited eojeol, so a
single wrong character fails a whole token and the number runs far above CER.

Note that `test` is deliberately left unfiltered (asr.sh stage 4 applies
`--min_wav_duration` to train/valid only), so it still contains the very short
backchannel utterances that the training set drops. Those are easy, which pulls
the test CER below what the training-time validation numbers suggest.

### Do not undo the small model and the duration floor

An earlier version of this recipe — a 12-block encoder, lr 2e-3, `ctc_weight`
0.3, no duration floor and no speed perturbation — is kept as
`conf/tuning/train_asr_scratch_v1.yaml`. It does **not** work: validation peaked
at epoch 9 and then overfitted for 38 more epochs (train acc 0.582 vs valid acc
0.277), `cer_ctc` never moved off 0.83, and decoding emitted frequent fillers
regardless of the audio ("네", "음", "어 좀 좀 좀 ..."). The cause was in the
data, not the model: 9.4% of the training utterances were a single backchannel
syllable (네 x1360, 음 x367, ...) and p10 of the token count was 1, so the loss
could be driven down by learning the label prior instead of the acoustics.
`conf/train_asr.yaml` carries the full reasoning in comments.

## Reference

Yun, Weonhee, Kyuchul Yoon, Sunwoo Park, Juhee Lee, Sungmoon Cho, Donghoon Kang,
Koonhyuk Byun, Hyeonzu Hahn and Jungsun Kim. 2015. "The Korean Corpus of
Spontaneous Speech." *Phonetics and Speech Sciences* 7(2), 103–109.
