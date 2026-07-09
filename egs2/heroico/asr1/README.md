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
model, that band scores 82.7% WER versus 3.0% WER on the clean remainder
(17.3% combined). These utterances are kept and scored, consistent with the
Kaldi recipe; treat the Recordings WER with this caveat in mind. Since prompt
ids 355-560 are excluded from training, this mislabeling does not contaminate
the training transcripts.

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
- date: `Mon Jul  6 01:25:47 UTC 2026`
- python version: `3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.11.0+cu128`
- Git hash: `ff67a6d20f235f69e0a31446e00eef646a293d1d`
  - Commit date: `Wed Jun 24 23:09:10 2026 +0000`

## exp/asr_train_asr_conformer_raw_es_char
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_asr_model_valid.acc.ave/dev|1510|11447|90.9|6.9|2.2|1.1|10.2|28.9|
|decode_asr_transformer_asr_model_valid.acc.ave/test|4636|22206|86.2|12.3|1.5|2.0|15.8|37.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_asr_model_valid.acc.ave/dev|1510|64085|96.8|1.1|2.1|0.9|4.1|28.9|
|decode_asr_transformer_asr_model_valid.acc.ave/test|4636|116655|95.4|2.6|1.9|1.4|6.0|37.6|
