# SLURP spoken language understanding

Two-pass SLU on [SLURP](https://github.com/pswietojanski/slurp), ported from
`egs2/slurp/slu1`. The system is `esp2_slu`
(`espnet3/systems/esp2_slu/`), which drives espnet2's `SLUTask`.

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

## Getting the corpus

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

A correct build gives 50,627 / 69,253 / 8,690 / 13,078 utterances and 69 intent
labels; the devel and test counts match `egs2/slurp/slu1/README.md`.

## Running it

The three training configs are stages of one pipeline, not alternatives: the
two-pass models start from the first pass's encoder, and the ASR-transcript one
also reads transcripts the first pass decoded.

```bash
. path.sh

# 1) Manifests and the intent list
python run.py --stages create_dataset \
    --training_config conf/tuning/training_conformer.yaml

# 2) First pass: a one-pass model that already predicts the intent
python run.py --stages collect_stats train \
    --training_config conf/tuning/training_conformer.yaml

# 3) Second pass, on the reference transcript
python run.py --stages train \
    --training_config conf/tuning/train_slu_bert_conformer_deliberation.yaml

# 4) Score it
python run.py --stages infer measure \
    --training_config conf/tuning/train_slu_bert_conformer_deliberation.yaml \
    --inference_config conf/inference_slu.yaml \
    --metrics_config conf/metrics.yaml
```

For the ASR-transcript variant, dump the first pass's hypotheses over every
split first, then train and score against those:

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

## Results

50 epochs each on one A40. `egs2/slurp/slu1` is the reference; it reports
intent accuracy only.

### Two-pass, reference transcript

`conf/tuning/train_slu_bert_conformer_deliberation.yaml`

| test set | intent accuracy | egs2 | transcript WER | transcript CER |
|---|---|---|---|---|
| devel | **90.98** | 89.6 | 6.24 | 4.29 |
| test | **90.45** | 89.0 | 6.41 | 4.39 |

### Two-pass, first-pass ASR transcript

`conf/tuning/train_slu_bert_conformer_deliberation_asr.yaml`

| test set | intent accuracy | egs2 | transcript WER | transcript CER |
|---|---|---|---|---|
| devel | 86.26 | 86.6 | 18.47 | 12.38 |
| test | 85.87 | 86.8 | 18.59 | 12.23 |

### First pass on its own

`conf/tuning/training_conformer.yaml`

| test set | intent accuracy | transcript WER | transcript CER |
|---|---|---|---|
| devel | 86.02 | 17.40 | 11.89 |
| test | 85.08 | 17.51 | 11.67 |

## Notes

- Training uses the real and the synthetic training splits together, as egs2
  does. Drop the `train_synthetic` entry from `dataset.train` to train on real
  speech only.
- Inference reproduces what egs2 actually ran: beam 20 with CTC weight 0.5.
  Its `run.sh` passes no `--inference_config`, so the espnet2 CLI defaults
  decided its published numbers.
- Decoding is beam-search bound rather than GPU bound -- about 8.6 s per
  utterance either way. Shard it with `parallel.n_workers`, which the shipped
  configs leave at 1.
- **Check the size of `valid.acc.ave_*.pth` after training.** Averaging writes
  the file even when it ends up empty, and an empty average decodes as a
  randomly initialised model: fluent-looking nonsense and near-zero intent
  accuracy. A healthy average here is ~1.2 GB.
- Samples hold `speech`, `text` and optionally `transcript`. SCP files are
  keyed by item index, because recipe samples must not carry a `utt_id` field.
