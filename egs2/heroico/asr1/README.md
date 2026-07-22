# Heroico Recipe

This is the ASR recipe of the [LDC West Point Heroico Spanish Speech corpus](https://www.openslr.org/39/) (LDC2006S37, Apache 2.0 license).

The corpus consists of three subsets of Latin American Spanish speech:
- **Heroico Answers**: spontaneous answers to questions, recorded at the Military Academy of Heroico in Mexico
- **Heroico Recordings**: read speech of prompted sentences, recorded at the Military Academy of Heroico in Mexico
- **USMA**: read speech of prompted sentences, recorded at the United States Military Academy (West Point)

## Data split

The split is speaker-disjoint (train: heroico speakers 1-82, dev: 83-92,
test: 93-102 plus all USMA speakers) and, for the read-speech subsets, also
text-disjoint:

- The Recordings prompt pool (`heroico-recordings.txt`, 724 sentences) mixes
  519 sentences from military-history lecture notes with 205 short "language
  learning" sentences (prompt ids 355-560). Each speaker reads only a slice of
  the pool, but the 82 train speakers jointly cover all of it, so a purely
  speaker-based split would leave every dev/test Recordings transcript also
  present verbatim in train.
- USMA speakers all recite the same 205 language-learning prompts
  (`usma-prompts.txt`), i.e. the id 355-560 pool.
- Therefore prompt ids 355-560 are held out for dev/test Recordings and are
  excluded from train Recordings, mirroring the Kaldi `egs/heroico` recipe
  (which routes the same id range to its devtest set). This keeps dev/test
  Recordings and USMA transcripts unseen in training.
- Answers is spontaneous speech and is split by speaker only. Roughly 28% of
  its dev/test utterances coincidentally match a short train phrase (e.g.
  "no estoy bien"); that residual overlap is inherent to natural language,
  unlike the structural overlap above.

### Interpreting the test WER

Because of the hold-out above, the test set is dominated by conditions the
model has never seen: 80% of test utterances are USMA (different recording
site, mostly non-native speakers, reading the held-out sentences), and the
Recordings portion is read speech over held-out text. Per-subset WER of the
released model:

| test subset | utts | WER |
|---|---|---|
| answers (spontaneous, novel speakers) | 711 | 18.9% |
| recordings, clean (held-out text) | 129 | 54.3% |
| recordings, mislabeled band (see below) | 55 | 98.4% |
| usma native | 1647 | 57.0% |
| usma non-native | 2028 | 71.3% |

The aggregate test WER is therefore mostly an out-of-domain read-speech
measure; the spontaneous in-domain measure is the answers row (and the dev
set, which is answers-only). CER (test 20.6%, dev 8.1%) is the better
indicator of acoustic quality for the read subsets, where errors are
dominated by phonetically-close word substitutions on unseen sentences.

## Known corpus label noise (Recordings)

Two bands of Recordings prompts (ids ~477-503 and ~528-555) are dialogue-style
lines such as "gracias" or "a qué hora sale el avión". Some speakers answered
these prompts conversationally instead of reading them (e.g. audio "no estoy
segura pero me parece que las cinco cuarenta de la tarde" against reference
"a qué hora sale el avión"), while `heroico-recordings.txt` always stores the
prompt text, so audio and reference do not match for those utterances. This is
a corpus-level issue: any split inherits it, and which utterances are affected
depends on speaker behavior (in this recipe's test set, speakers 097-100
answered ~55 of the 184 Recordings utterances). Measured with the released
model, that band scores 98.4% WER versus 54.3% WER on the clean remainder,
whose own errors are dominated by the held-out-text condition. These
utterances are kept and scored, consistent with the Kaldi recipe; treat the
Recordings WER with this caveat in mind. Since prompt ids 355-560 are
excluded from training, this mislabeling does not contaminate the training
transcripts.

## How to run

```sh
bash run.sh
```

```
@misc{LDC2006S37,
	Author = {Morgan, John},
	Title = {West Point Heroico Spanish Speech},
	Publisher = {Linguistic Data Consortium},
	Address = {Philadelphia},
	Year = {2006},
	Note = {LDC Catalog No. LDC2006S37}}
```


# RESULTS
## Environments
- date: `Mon Jul 13 05:30:31 EDT 2026`
- python version: `3.10.20 | packaged by conda-forge | (main, Jun 11 2026, 03:31:56) [GCC 14.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.9.1+cu128`
- Git hash: `c4d0e10a4ab07dec3f3daaaabe3dd840fa23f38f`
  - Commit date: `Sun Jul 12 14:22:26 2026 -0400`

## Results
- ASR config: [conf/train_asr_conformer.yaml](conf/train_asr_conformer.yaml)
- Decode config: [conf/decode_asr_transformer.yaml](conf/decode_asr_transformer.yaml)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_asr_model_valid.acc.ave/dev|919|5207|77.9|18.3|3.8|2.3|24.4|49.6|
|decode_asr_transformer_asr_model_valid.acc.ave/test|4570|21651|50.5|44.4|5.1|6.8|56.3|85.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_asr_model_valid.acc.ave/dev|919|26361|93.5|3.5|3.0|1.6|8.1|49.6|
|decode_asr_transformer_asr_model_valid.acc.ave/test|4570|113280|84.0|10.5|5.5|4.6|20.6|85.6|
