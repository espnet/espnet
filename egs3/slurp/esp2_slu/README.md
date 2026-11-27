# SLURP SLU recipe

Two-pass spoken language understanding on
[SLURP](https://github.com/pswietojanski/slurp), ported from
`egs2/slurp/slu1`. The system is `esp2_slu` (`espnet3/systems/esp2_slu/`),
which drives espnet2's `SLUTask`; this directory holds only what is specific
to the corpus -- its manifests, its configs and its tokenizer hooks.

The target text is the intent label followed by the transcript:

```
audio_volume_mute turn off the speakers
^ scenario_action ^ transcript
```

The first predicted token is therefore the intent, scored as classification
accuracy; the rest is scored as WER/CER. Intent labels are reserved as
SentencePiece symbols, so each stays a single token.

The second pass additionally reads a transcript, which a frozen BERT
post-decoder encodes and a deliberation encoder fuses with the speech encoder
output. That transcript is either the reference or the first pass's own
hypothesis -- the two settings egs2 selects with `--gt true` / `--gt false`.

## Data Preparation

SLURP is not downloaded by the recipe: the metadata is a git repository and the
audio is fetched by a script inside it. Both are needed.

```bash
# 1) metadata (jsonl + metadata.json)
git clone https://github.com/pswietojanski/slurp.git download/slurp

# 2) audio (real and synthetic, ~58 GB unpacked)
cd download/slurp && bash scripts/download_audio.sh
```

Expected layout:

```
download/slurp/
├── dataset/slurp/{metadata.json,train.jsonl,train_synthetic.jsonl,devel.jsonl,test.jsonl}
└── audio/{slurp_real,slurp_synth}/*.flac
```

If the corpus lives elsewhere, export `SLURP=/path/to/slurp` (the variable
`egs2/TEMPLATE/asr1/db.sh` uses) or set `create_dataset.source_dir` in the
training config. A symlink at `download/slurp` works too.

`create_dataset` then writes flat manifests under `data/manifest/`:

| file | content |
|---|---|
| `train.tsv`, `train_synthetic.tsv`, `devel.tsv`, `test.tsv` | `utt_id`, `wav_path`, `intent`, `transcript` |
| `intents.txt` | intent labels seen in the training splits |

| split | utterances |
|---|---|
| train | 50,627 |
| train_synthetic | 69,253 |
| devel | 8,690 |
| test | 13,078 |

A correct build also gives 69 intent labels; the devel and test counts match
`egs2/slurp/slu1/README.md`.

A sample holds `speech`, `text`, `intent` and `reference_transcript`, plus
`transcript` when a config asks for it. That last one is the model's *input*
-- the reference or a first-pass hypothesis -- while `reference_transcript` is
always the corpus text, which is what `conf/metrics.yaml` scores against with
its `dataset:` bindings. SCP files are keyed by item index, because recipe
samples must not carry a `utt_id` field.

## Commands

Run from this directory, after `. path.sh`. The three configs are stages of one
pipeline rather than alternatives: both two-pass models start from the first
pass's encoder, and the ASR-transcript one also reads transcripts it decoded.

### Conformer (first pass)

A one-pass model that already predicts the intent, and the encoder the two-pass
models start from.

```bash
# 1) Manifests and the intent list
python run.py --stages create_dataset \
    --training_config conf/tuning/training_conformer.yaml

# 2) Train
python run.py --stages collect_stats train \
    --training_config conf/tuning/training_conformer.yaml

# 3) Decode and score
python run.py --stages infer measure \
    --training_config conf/tuning/training_conformer.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

### Conformer deliberation, reference transcript

```bash
python run.py --stages train \
    --training_config conf/tuning/train_slu_bert_conformer_deliberation.yaml

python run.py --stages infer measure \
    --training_config conf/tuning/train_slu_bert_conformer_deliberation.yaml \
    --inference_config conf/inference_slu.yaml \
    --metrics_config conf/metrics.yaml
```

### Conformer deliberation, first-pass transcript

Dump the first pass's hypotheses over every split first, then train and score
against those.

```bash
python run.py --stages infer \
    --training_config conf/tuning/training_conformer.yaml \
    --inference_config conf/inference_transcripts.yaml

python run.py --stages train \
    --training_config conf/tuning/train_slu_bert_conformer_deliberation_asr.yaml

python run.py --stages infer measure \
    --training_config conf/tuning/train_slu_bert_conformer_deliberation_asr.yaml \
    --inference_config conf/inference_slu_asr.yaml \
    --metrics_config conf/metrics.yaml
```

`measure` writes `metrics.json` next to the inference output, plus
`intent_errors` (the misclassified utterances) and `wer_alignment` per test set.

Decoding is beam-search bound rather than GPU bound -- about 8.6 s per
utterance either way. Shard it with `parallel.n_workers`, which the shipped
configs leave at 1.

## Results

50 epochs each on one A40. `egs2/slurp/slu1` is the reference; it reports
intent accuracy only.

### Conformer (first pass)

- training: `conf/tuning/training_conformer.yaml`
- inference: `conf/inference.yaml`
- metrics: `conf/metrics.yaml`

| test set | intent accuracy | transcript WER | transcript CER |
|---|---|---|---|
| devel | 86.02 | 17.40 | 11.89 |
| test | 85.08 | 17.51 | 11.67 |

`conf/inference.yaml` and `conf/inference_transcripts.yaml` decode this model,
so they swap in `FirstPassTranscriptRunner`: an ASR hypothesis is words from
its first token, and the SLU runner would strip that word as an intent label.

### Conformer deliberation, reference transcript

- training: `conf/tuning/train_slu_bert_conformer_deliberation.yaml`
- inference: `conf/inference_slu.yaml`
- metrics: `conf/metrics.yaml`

| test set | intent accuracy | egs2 | transcript WER | transcript CER |
|---|---|---|---|---|
| devel | **90.98** | 89.6 | 6.24 | 4.29 |
| test | **90.45** | 89.0 | 6.41 | 4.39 |

### Conformer deliberation, first-pass transcript

- training: `conf/tuning/train_slu_bert_conformer_deliberation_asr.yaml`
- inference: `conf/inference_slu_asr.yaml`
- metrics: `conf/metrics.yaml`

| test set | intent accuracy | egs2 | transcript WER | transcript CER |
|---|---|---|---|---|
| devel | 86.26 | 86.6 | 18.47 | 12.38 |
| test | 85.87 | 86.8 | 18.59 | 12.23 |

#### Comparison with egs2

The reference-transcript model is 1.4 points above `egs2/slurp/slu1` on test;
the first-pass-transcript one is 0.9 below. Data preparation and model
architecture match egs2 field for field, down to the quirk that only SLURP's
synthetic split is lowercased. Two things differ deliberately:

- **Vocabulary size.** egs2 tokenizes offline in `local/run_spm.sh`: BPE-500
  over the transcripts, the text rewritten as `<intent> <bpe> <pieces>`, then
  `--token_type word`, so the intent survives because the final vocabulary is a
  word list over already-encoded text (~570 tokens). This does it in one pass
  with the intents as `user_defined_symbols`, giving 999 tokens.
- **Frontend.** egs2 sets no `frontend_conf` and gets `DefaultFrontend`'s
  32 ms / 8 ms; this uses the 25 ms / 10 ms the other egs3 recipes set.

Training uses the real and the synthetic splits together, as egs2 does; drop
the `train_synthetic` entry from `dataset.train` to train on real speech only.
Inference reproduces what egs2 actually ran, beam 20 with CTC weight 0.5: its
`run.sh` passes no `--inference_config`, so the espnet2 CLI defaults decided
its published numbers.
