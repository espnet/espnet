# Seoul Corpus ASR recipe

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

* **The corpus's own utterance boundaries are kept.** Every annotated interval
  that carries speech becomes one utterance, exactly as the corpus segments it.
  The tier breaks at every pause, so they are short — 46180 intervals averaging
  1.56 s in train.

* **Only the training and validation sets are length-filtered.** `test` is built
  with no duration filter at all, so every utterance the corpus annotates is
  scored — 5899 of them, down to 0.07 s. Both filters that could touch it are
  disabled for `test`: `local/data.sh` passes `--min_duration 0` for that part,
  and asr.sh stage 4 already skips `test_sets` by design. Train and valid keep
  `--min_wav_duration 1.0`, which drops 26% of the utterances but only 5% of the
  audio and is what stops the model learning the label prior instead of the
  acoustics (see the note under Baseline).

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

Measured on the full `test` set — all 5899 annotated utterances, nothing merged
and nothing filtered out — decoding the average of the 10 best checkpoints after
40 epochs (432 batches/epoch, ~17.3k steps):

| | Corr | Sub | Del | Ins | **Err** |
| --- | --- | --- | --- | --- | --- |
| **CER** | 85.2 | 10.5 | 4.4 | 2.7 | **17.6** |
| WER | 68.3 | 26.7 | 5.0 | 2.8 | 34.5 |
| TER | 74.8 | 18.0 | 7.2 | 2.1 | 27.3 |

Report **CER**: Korean WER is computed over whitespace-delimited eojeol, so a
single wrong character fails a whole token and the number runs far above CER.

The short end of that test set is what makes the number what it is: a quarter of
the utterances are under half a second, fragments such as 근데 / 아 / 네 with
almost no context to condition on. Applying the training set's 1.0 s floor to
test as well would report a visibly better CER while measuring less, which is why
this recipe does not.

### Do not undo the duration floor on the training data

An earlier version of this recipe — a 12-block encoder, lr 2e-3, `ctc_weight`
0.3, no duration floor and no speed perturbation — did **not** work: validation
peaked at epoch 9 and then overfitted for 38 more epochs (train acc 0.582 vs
valid acc 0.277), `cer_ctc` never moved off 0.83, and decoding emitted frequent
fillers regardless of the audio ("네", "음", "어 좀 좀 좀 ..."). The cause was in
the data, not the model: 9.4% of the training utterances were a single
backchannel syllable and p10 of the token count was 1, so the loss could be
driven down by learning the label prior instead of the acoustics. Keep
`--min_wav_duration` at 1.0 for train/valid; it is what breaks that shortcut
(the most frequent transcript drops from 4.4% of the training set to 0.1%).

## Reference

Yun, Weonhee, Kyuchul Yoon, Sunwoo Park, Juhee Lee, Sungmoon Cho, Donghoon Kang,
Koonhyuk Byun, Hyeonzu Hahn and Jungsun Kim. 2015. "The Korean Corpus of
Spontaneous Speech." *Phonetics and Speech Sciences* 7(2), 103–109.
