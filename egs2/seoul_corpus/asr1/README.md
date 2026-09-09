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

* **The corpus's own utterance boundaries are kept.** The tier breaks at every
  pause, so the utterances are short — 45953 fragments averaging 1.57 s in train.
  `--merge_gap` can glue neighbours separated by nothing but silence/noise back
  together, but it defaults to 0, i.e. off; see "Merging is available and off by
  default" below for what turning it on costs.

* **Only the training and validation sets are length-filtered.** `test` is built
  with no duration filter at all, so every utterance the corpus annotates is
  scored — 5595 of them, down to 0.07 s. Both filters that could touch it are
  disabled for `test`: `local/data.sh` passes `--min_duration 0` for that part,
  and asr.sh stage 4 already skips `test_sets` by design. Train and valid keep
  `--min_wav_duration 1.0`, which drops 26% of the utterances but only 5% of the
  audio and is what stops the model learning the label prior instead of the
  acoustics (see the note under Baseline).

## Merging is available and off by default

`--merge_gap` does not merely tidy the data up; it redefines what an utterance
is, and `local/data.sh` would apply it to **all three sets, test included**. Turn
it on and the scores no longer sit on the segmentation the corpus ships, so they
stop being comparable with anything measured on it — including the baseline
below. It is off by default for that reason.

Measured at `--merge_gap 0.5`, for reference:

| set | fragments | utterances out | | audio | merged utterances | audio inside them |
| --- | --- | --- | --- | --- | --- | --- |
| train | 45953 | 26446 | −42.4% | 20.0 h → 21.7 h | 10490 (39.7%) | 68.2% |
| dev | 5951 | 3722 | −37.5% | 2.3 h → 2.5 h | 1353 (36.4%) | 62.8% |
| test | 5595 | 3164 | −43.4% | 2.1 h → 2.4 h | 1216 (38.4%) | 72.5% |

So about 40% of the utterances would be merges of two or more annotated
fragments, holding roughly 70% of the audio, and the total duration goes *up*
because merging also absorbs the silence between fragments (1.6 h in train).

Duration **filtering** is routine and ESPnet ships it (asr.sh stage 4), and
**splitting** over-long segments is a well-trodden recipe step (`egs2/ami/asr1`
does it from punctuation). Gap-based **merging of already-annotated utterances**
is not something another `egs2` recipe does, which is the other reason it is not
the default here. It measured about the same either way: 17.1 CER on the shipped
segmentation against 16.1 on the merged one, on test sets that are not the same
size, so treat that as "no clear difference" rather than a ranking.

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

Measured on the full `test` set — all 5595 annotated utterances, 65941
characters, nothing merged and nothing filtered out — decoding the average of the
10 best checkpoints after 40 epochs (440 batches/epoch, ~17.6k steps):

| | Corr | Sub | Del | Ins | **Err** |
| --- | --- | --- | --- | --- | --- |
| **CER** | 85.6 | 10.2 | 4.2 | 2.7 | **17.1** |
| WER | 68.7 | 26.4 | 4.9 | 2.8 | 34.1 |
| TER | 75.3 | 17.8 | 7.0 | 2.1 | 26.8 |

Report **CER**: Korean WER is computed over whitespace-delimited eojeol, so a
single wrong character fails a whole token and the number runs far above CER.

Keeping the sub-second utterances in the test set costs about 0.8 CER: filtering
them out the way the training data does gives 16.3 over 5379 utterances. They are
0.07-0.2 s fragments such as 근데 / 아 / 네 with almost no context, so dropping
them flatters the score rather than measuring anything.

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
