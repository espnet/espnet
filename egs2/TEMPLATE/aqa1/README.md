# Audio quality assessment (AQA)

This task trains predictors of speech/audio evaluation metrics. `aqa`
is the task name; Uni-VERSA is the initial model (`universa: base`).

The implementation revives [ESPnet PR #5959](https://github.com/espnet/espnet/pull/5959)
on current ESPnet. It uses the existing length-aware speaker pooling API.

## Data

Each split needs the usual `wav.scp`, `utt2spk`, and `spk2utt`, plus a
`metric.scp` with an utterance ID followed by a JSON object:

```text
utt1 {"utmos": 3.5, "pesq": 2.1, "wer": 10.0}
utt2 {"utmos": 2.8, "wer": 15.0}
```

Uni-VERSA regression values must be numeric. ARECHO also supports categorical
values through its published metric tokenizer. Missing labels are omitted from the JSON object;
the collator masks them with `metric_pad_value` (default `-100`). A batch with
no labels for a metric contributes zero loss for that metric. Values at or below
the padding threshold are treated as missing.

Generate the target scores with the corresponding evaluation tools (for example,
VERSA), then serialize their utterance-level results into this keyed JSON format.
The recipe consumes these scores; it does not generate training labels itself.
Synthetic metric labels should be used for integration testing only.

Optional inputs are `ref_wav.scp` and `text`. Disable them with
`--use_ref_wav false` and `--use_ref_text false`. A missing reference-audio entry
may use `utt_id None`; preprocessing supplies silence for that entry.

The recipe discovers the metric vocabulary from training `metric.scp`. You can
also supply `--metric2id path/to/names`, containing one metric name per line.
The saved model config embeds those names for portable inference.

## Run

Create a recipe under `egs2/<corpus>/aqa1` with the shared setup script:

```bash
./egs2/TEMPLATE/aqa1/setup.sh egs2/my_corpus/aqa1
cd egs2/my_corpus/aqa1
# Implement local/data.sh to prepare the splits described above, then run:
./aqa.sh --ngpu 0 --train_set train --valid_set dev --test_sets test
```

Setup copies `cmd.sh`, `conf`, and `local` from the ASR template so each recipe
can customize them. It links the common runtime scripts to `TEMPLATE/asr1` and
the task driver to `TEMPLATE/aqa1/aqa.sh`. Run setup from the repository root;
the target must follow the `egs2/<corpus>/aqa1` layout.

Stages prepare data, format audio, discover metrics, filter data, tokenize text,
collect statistics, train, infer, and score. Supply a training configuration with
`--train_config path/to/train.yaml` for your model and corpus.

Python entry points are `espnet2.bin.aqa_train` and
`espnet2.bin.aqa_inference`; model packing uses `espnet2.bin.pack aqa`.
Old checkpoint state-dict keys retain the `universa` model prefix.

## ARECHO and legacy models

Select `universa: ar_universa`, set `sequential_metric: true`, and provide `metric_token_info` (the published
JSON tokenizer) and `metric2type` for autoregressive training. Saved configs
embed this metadata. Inference supports `--beam_size`, `--skip_meta_label_score`,
`--save_token_seq`, and `--use_fixed_order --fixed_metric_name_order language,srmr`.
ARECHO inference currently accepts one utterance per batch.

The old `espnet2.bin.universa_train` / `universa_inference` entry points
remain aliases for AQA. Parameter names and
published token IDs are preserved so existing weights can initialize training.
Historical `embedding_dim` and `use_rope` fields were ignored by the original
models; they remain ignored to preserve checkpoint shapes and positional encoding.
Use `embedding_size` to configure new models.

For fine-tuning, resolve paths in the original config to the downloaded metric
vocabulary, tokenizer, and optional BPE model, then train with that config and
`--init_param /path/to/model.pth`. Keep the published `freeze_param` setting if
you intend to freeze the same modules. `--resume true` instead requires a full
ESPnet trainer checkpoint (including optimizer and training state).
