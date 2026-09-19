# VoxLingua107 LID recipe

Set `VOXLINGUA107` to the corpus destination, or set `dataset_dir` in both the
training and inference configs. The environment variable also supplies the
DatasetBuilder's default source path. The fallback `/path/to/voxlingua107` is a
placeholder and must be changed before preparation.
`create_dataset` now downloads the official ZIPs and extracts them, following
ESPnet2 `local/data.sh`: obtain `zip_urls.txt`, download with `wget --continue`,
and extract with `unzip -q -o`. The official list contains only training ZIPs,
so `dev.zip` is downloaded explicitly. Install `wget` and `unzip` first.
Training audio goes under `<dataset_dir>/<lang>/` and development audio under
`<dataset_dir>/dev/<lang>/`. Audio bytes and utterance lengths are preserved.
Already prepared sources are reused. Interrupted downloads resume; interrupted
extraction is retried and ZIP CRC errors stop preparation. The downloaded URL
list is retained across retries. Source URLs can be overridden with
`create_dataset.zip_urls_url` and `create_dataset.dev_zip_url` (defaults live in
`dataset/config.yaml`). This path downloads official ZIPs; it does not read
WebDataset tar files.
Dataset construction never downloads audio or builds manifests. Run
`create_dataset` explicitly to prepare the corpus and manifests. Dataset readers
use only the requested split's manifest and its referenced audio. Evaluating dev
does not require training audio or training manifests; inference still needs
the model's training `lang2utt` to interpret class IDs. Interrupted manifest
builds are marked incomplete and rebuilt by the preparation stage.

Like the manifest-building ASR recipe (`mini_an4`), generated data stays in the
recipe, separate from the source corpus:

```text
data/voxlingua107/{train,dev}/  # manifest.tsv and source-local label mappings
exp/stats/{train,valid}/       # speech_shape and combined-dataset label mappings
exp/<exp_tag>/                # checkpoints and inference results
```

Audio is read directly from the source; it is not copied to `dump/`. Builder and
Dataset use `data_dir: ${data_dir}/voxlingua107` for generated manifests. The
default `data_dir` is `${recipe_dir}/data`. Existing `<source>/espnet3/` files are
left untouched and are no longer used by the shipped configs. Manifest audio
paths remain absolute: after relocating audio, rebuild manifests in a fresh
`data_dir` and rerun `collect_stats` before training. If overriding `data_dir` or
`stats_dir`, use the same paths in training, inference, and publication configs.

For multiple training datasets, add entries to `dataset.train` (and `valid`),
then rerun `collect_stats`. The LID collector writes `category2utt` and `lang2utt`
with the same combined indices as `speech_shape`; the sampler and preprocessor
read these combined mappings, not one corpus's local indices. Distinct source
corpora need distinct manifest directories. Raw samples must have a string
`lid_labels` language code; update `model.lang_num` if the language inventory
changes. Changing dataset order also requires recollecting statistics.

The collector also writes `dataset2utt`/`utt2dataset`, with source positions
(`0`, `1`, ...) as dataset IDs. To use `catpow_balance_dataset`, point each split's
`batches.dataset2utt_parent_dir` at `${stats_dir}/train` or `${stats_dir}/valid`
and set `category_upsampling_factor` and `dataset_upsampling_factor`.

The template keeps an integer iterator seed of `0` when top-level `seed` is
unset. This recipe explicitly passes `${seed}` to both iterators, so its default
`3702` controls their shuffle and worker seeds too. This changes data ordering
relative to older configs that implicitly used iterator seed `0`.

## Training schedule and ESPnet2 compatibility

The default is 30 epochs with 1,000 sampled batches per epoch and gradient
accumulation over 2 batches: 30,000 input batches and 15,000 optimizer updates
when none are skipped. An epoch is not a complete pass through the corpus.
The category sampler and iterator batch selection match ESPnet2 for the same
single-GPU inputs, seed, and epoch. This does not establish identical numerical
training trajectories. On BF16-capable GPUs such as H200, both this recipe and
the current ESPnet2 trainer use BF16 autocast. ESPnet2 falls back to FP16 on
other GPUs; use `trainer.precision: 16-mixed` on those devices.

The copied ESPnet2 scheduler setting, `max_steps: 30000`, counts optimizer
updates. Its 9,000 warmup and 6,000 hold updates occupy almost the entire run,
so the configured final learning rate is not reached. To complete all three
phases in 30 epochs, override `scheduler.max_steps` to `15000`; that is a change
to the ESPnet2 baseline, and should be reported as such.

Validation also retains the ESPnet2 category sampler. Use the separate `infer`
and `measure` stages to score every prepared development utterance once.
Like ESPnet2, validation runs after each training epoch, with no additional
sanity-validation batches before the first epoch. cuDNN benchmark and
deterministic settings also follow the ESPnet2 baseline.
The LID precision callback restores FP32 matmul precision to `highest`, matching
ESPnet2's `use_tf32: false`; the common ESPnet3 entrypoint otherwise sets `high`.
This affects FP32 matrix multiplication, while BF16 autocast remains enabled.

Inference selects the best individual checkpoint by validation accuracy,
matching ESPnet2 `run.sh`. The LID callback creates
`valid.accuracy.best.pth` as a relative link after each epoch's validation, so
saved models can also be evaluated before training finishes.
The common top-2 retention and averaging remain available; explicitly select
`valid.accuracy.ave_2best.pth` to evaluate the averaged model instead.

With multiple GPUs, ESPnet3 assigns
whole batches to ranks, whereas ESPnet2's category path splits each batch
between ranks. The same configuration therefore does not imply the same
global batch size or sample weighting.

Inference must use the training `lang2utt` in its original class order.

## Run the stages

```bash
export VOXLINGUA107=/path/to/voxlingua107
python run.py --stages create_dataset --training_config conf/training.yaml
python run.py --stages collect_stats --training_config conf/training.yaml
python run.py --stages train --training_config conf/training.yaml
python run.py --stages infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
python run.py --stages measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
python run.py --stages pack_model \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml \
  --publication_config conf/publication.yaml
python run.py --stages upload_model \
  --training_config conf/training.yaml \
  --publication_config conf/publication.yaml
```

## Additional corpora

[Raw-corpus preparation](prepare_corpora.md) covers FLEURS, ML-SUPERB 2.0,
VoxPopuli downloads, and locally obtained Babel audio/transcripts. The resulting
manifests can be used for evaluation or combined training.

## Embeddings and t-SNE

Label-only inference remains the default. Enable optional normalized language
embeddings and t-SNE through the existing infer and measure stages:

```bash
python run.py --stages infer measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference_embeddings.yaml \
  --metrics_config conf/metrics_embeddings.yaml
```

Outputs are separate from label-only inference, under
`${exp_dir}/inference_embeddings/<test_name>/`:

- `embedding.scp` and per-utterance `.npy` files;
- `<test_name>_lang_to_embds.npz` and normalized language means in
  `<test_name>_lang_to_avg_embd.npz`;
- `tsne_plots/`, generated by the existing ESPnet2 plotting function.

The recipe summarizes at most 100 utterances per reference language, matching
the ESPnet2 VoxLingua setting. Unlike its inference-time cap, this cap applies
after full inference, only to embedding summaries and plots. Set the metric's
`inputs.ref: hyp` to group embeddings by predicted languages instead.
Perplexity is capped below the number of plotted points for small datasets.
Plotting dependencies are loaded only when requested: matplotlib/pandas for
PNG/CSV, optional plotly for HTML and adjustText for label positioning.

Standard measure also writes `lid_per_language.json` and
`lid_error_counts.json`. Existing scalar metrics and `lid_errors` are retained.

## Differences from the ESPnet2 LID template

This recipe covers official VoxLingua107 preparation, statistics, training,
language prediction, embeddings, scoring, t-SNE, and ESPnet3 model publication.
It does not port every preprocessing stage of `egs2/TEMPLATE/lid1`:

- Speed perturbation expands training samples at read time (see below); it
  does not materialize separate WAV files as in the ESPnet2 formatting stage.
- Audio format conversion, resampling, and Kaldi `segments` extraction require
  separate preparation, such as ESPnet2's `pyscripts/audio/format_wav_scp.py`.
  The stock builder reads the official WAV tree directly.
- Inference writes `hyp.scp` / `ref.scp` and optional per-utterance `.npy` files.
  Completed inference shards can be reused; partially completed shards are
  rerun. ESPnet2's per-utterance inference checkpoint/resume is not provided.
- `pack_model` creates an ESPnet3 bundle directory, not the ESPnet2 ZIP format.

## Optional speed perturbation

In `conf/training.yaml`, set the training Dataset's
`data_src_args.speed_perturb_factors` to `[0.9, 1.0, 1.1]`, then rerun
`collect_stats` before `train`. The default `[1.0]` keeps the unaugmented baseline.
Each original utterance contributes one training sample per factor, with the
same language label. Development and inference audio are unchanged.

The recipe reuses `espnet2.layers.augmentation.speed_perturb` while reading audio;
original WAVs and manifests are preserved. Statistics include the actual lengths
and category IDs of all speed variants. This reproduces the three-variant data
expansion, using ESPnet's torchaudio resampler rather than ESPnet2's offline SoX
conversion; the resampled waveform values need not be bit-identical.

With three factors, the training Dataset has three times as many samples.
The configured 1,000 sampled batches per epoch and 30 epochs remain unchanged.
When switching factors, use a separate experiment or regenerate its statistics;
statistics from a different factor list must not be reused. Keep the same
`stats_dir` in training, inference, and publication configs if overriding it.
