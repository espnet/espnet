# LibriTTS Codec Recipe (ESPnet3)

## Repo scope

All work on this recipe (commits, branches, pushes, PRs) must happen in the
user's own fork, `NewGamezzz/espnet` (git remote `origin`), not the official
`espnet/espnet` repo (git remote `upstream`). Never push branches or open
PRs against `upstream` for this work.

## Goal

Port `espnet2/tasks/gan_codec.py` into an ESPnet3 recipe and train a GAN-based
neural codec (starting with Encodec) on LibriTTS.

This recipe follows the standard ESPnet3 stage-based recipe pattern used
elsewhere in `egs3/` (see `egs3/mini_an4/asr` and `egs3/TEMPLATE/asr` for the
reference layout).

## Status

Implemented: dataset builder, training config, `CodecSystem` (train /
collect_stats), shared GAN adapter, `egs3/TEMPLATE/codec` (generic
run.py/training.yaml/inference.yaml/metrics.yaml scaffold).
Deferred: `infer` / `measure` (see "Inference and metrics" below) — the
recipe currently only supports `create_dataset` / `collect_stats` / `train`.

`CodecSystem` and the shared GAN adapter (`GANLightningModule`,
`GANLightningTrainer`/`build_gan_trainer`) were promoted out of this recipe
into `espnet3/systems/codec/` (mirroring `espnet3/systems/asr/`) so future
GAN-codec recipes can reuse them without duplication — this recipe's own
`run.py` is now a thin wrapper importing `CodecSystem` and the generic
`build_parser`/`main` from `egs3.TEMPLATE.codec.run` (see "Default config
template" below). No local unit tests are kept in this recipe/branch;
verification here was end-to-end manual smoke testing (dataset builder ->
manifest, dataset `__getitem__`, real Encodec model forward pass via
`get_espnet_model`, and the chunk-based dataloader wired through
`CombinedDataset`) run against the actual torch/espnet2 environment. The
smoke test surfaced real bugs — all fixed at the recipe level; see "Worked
around: hyphenated utt_ids" below for the one that needed the most care.

## Decision: no extra stages

The codec task has no text, no tokenizer, no speaker embedding, and no extra
offline pipeline step (unlike ASR's `train_tokenizer` or the reference
VITS/TTS recipe's `compute_xvectors` / `remove_long_short` /
`create_token_list`). Confirmed from `espnet2/tasks/gan_codec.py`:

- `required_data_names()` returns only `("audio",)`.
- `build_preprocess_fn()` uses `CommonPreprocessor(token_type=None,
  speech_name="audio", force_single_channel=True, audio_pad_value=0.0)`.
- `build_collate_fn()` is a plain `CommonCollateFn`.

So this recipe uses only the default ESPnet3 stages:
`create_dataset -> collect_stats -> train -> infer -> measure`.
`CodecSystem` (`espnet3/systems/codec/system.py`) only customizes
`train`/`collect_stats` internally (to select the GAN trainer) — that is
not "adding a stage".

## Batching: chunk-based, not numel

Checked every real `egs2/*/codec1` recipe (libritts, audioset, musdb18,
amuse, mini_an4 — encodec/dac/soundstream/hificodec/funcodec configs): none
use numel/`batch_bins`. All use `batch_type: unsorted` + `iterator_type:
chunk` (fixed-length audio windows). `egs2/libritts/codec1/conf/tuning/
train_encodec.yaml` uses `chunk_length: 61440`, `batch_size: 4`,
`num_cache_chunks: 64` — these exact values are carried into
`conf/training_encodec.yaml`.

Consequence: `ESPnetGANCodecModel.collect_feats()` returning `{}`
unconditionally (`espnet2/gan_codec/espnet_model.py`) does **not** need a
patch. `espnet2.iterators.chunk_iter_factory.ChunkIterFactory` always builds
its per-sample sampler as `UnsortedBatchSampler(batch_size=1, key_file=...)`
regardless of `batch_type` (confirmed in `espnet2/tasks/abs_task.py`'s
`build_chunk_iter_factory`), and `UnsortedBatchSampler` only needs a key file
listing utt-ids — the manifest TSV itself works directly as that key file,
no `collect_stats` shape files required at all. If a future recipe wants
numel/length-based batching instead, the `collect_feats` patch (mirroring
`_patch_gan_tts_collect_feats` in the reference TTS branch) would become
necessary again — it isn't currently, so it isn't implemented.

## GAN adapter: shared, not recipe-local

`AbsGANESPnetModel.forward()` (which `ESPnetGANCodecModel` implements) needs
two forward passes per batch (`forward_generator=True/False`) and returns a
dict with `optim_idx`, which is incompatible with
`ESPnetLightningModule._step`'s single-call `(loss, stats, weight)` tuple
contract. This adapter is generic to any `AbsGANESPnetModel` (codec and
GAN-TTS both use it), so it lives under `espnet3/systems/codec/` (promoted
there from `espnet3/components/`, alongside `CodecSystem`) rather than
under `egs3/libritts/codec/`:

- `espnet3/systems/codec/gan_lightning_module.py` —
  `GANLightningModule(ESPnetLightningModule)`. Sets
  `automatic_optimization = False`; overrides `_step()` to loop over
  `("discriminator", "generator")` (order via `trainer.gan.generator_first`),
  calling `self.model(**batch[1], forward_generator=...)` twice per batch,
  normalizing `optim_idx`, and manually driving backward/optimizer-step/
  grad-clip/scheduler-step per named optimizer. Supports
  `trainer.gan.skip_discriminator_prob` (DDP-synced per-batch discriminator
  skip).
- `espnet3/systems/codec/gan_trainer.py` —
  `GANLightningTrainer(ESPnet3LightningTrainer)` strips the `gan:` sub-key
  out of `trainer:` config before it reaches plain Lightning's
  `Trainer(**trainer_config)`. Plus `build_gan_trainer(training_config,
  model)` factory.

Ported from `NewGamezzz/espnet:libritts_vits/egs3/libritts/tts/src/
{gan_trainer.py, models/gan_model.py}` (`GANTTSLightningModule` /
`GANTTSLightningTrainer`), renamed and generalized — the TTS-specific
`_patch_gan_tts_collect_feats` helper was dropped since it isn't needed here
(see batching section above). If/when a TTS system lands in this repo, it
should import these same shared classes instead of duplicating them.

`CodecSystem._build_trainer()` (`espnet3/systems/codec/system.py`) picks
between `build_gan_trainer` and plain `ESPnet3LightningTrainer` based on
`isinstance(model, AbsGANESPnetModel)`.

## Default config template

`egs3/TEMPLATE/codec/` mirrors `egs3/TEMPLATE/asr/`'s role and structure in
full, not just `training.yaml`: a generic `run.py` (`build_parser`/`main`
parameterized by `system_cls`, `DEFAULT_STAGES` without `train_tokenizer`)
plus `conf/training.yaml` / `conf/inference.yaml` / `conf/metrics.yaml`.
`training.yaml` is adapted for codec: no tokenizer section,
`optimizers`/`schedulers` (named multi-optimizer GAN path) active by
default instead of the ASR template's single-optimizer path, a `trainer.gan`
block, and a chunk-based `dataloader` scaffold (matching the batching
decision above) instead of the ASR template's numel-oriented one.
`inference.yaml`/`metrics.yaml` are placeholder scaffolds (see "Inference
and metrics" below).

`egs3/libritts/codec/run.py` is now a thin wrapper (mirroring
`egs3/mini_an4/asr/run.py`): it imports `DEFAULT_STAGES`/`build_parser`/
`main` from `egs3.TEMPLATE.codec.run` and passes `CodecSystem` as
`system_cls`. All three of `--training_config`/`--inference_config`/
`--metrics_config` are merged against `egs3.TEMPLATE.codec`'s own
`conf/{training,inference,metrics}.yaml` (`default_package=__package__`
inside `egs3/TEMPLATE/codec/run.py`, which resolves to
`egs3.TEMPLATE.codec` regardless of which recipe's `run.py` calls `main()`
— see `espnet3.utils.config_utils.load_and_merge_config`). This is the same
layered-defaults pattern `egs3/mini_an4/asr` uses against
`egs3/TEMPLATE/asr`: recipe-specific yaml files only need to specify what
differs from the generic scaffold (e.g. `training_encodec.yaml` doesn't
define `parallel:`, and inherits `{env: local, n_workers: 1}` from the
template instead of leaving it unset).

## Dataset

`dataset/builder.py` (`LibriTTSCodecBuilder`) / `dataset/dataset.py`
(`LibriTTSCodecDataset`) mirror the manifest-scanning pattern from
`egs3/libritts/tts/dataset/builder.py` (`NewGamezzz/espnet:libritts_vits`),
stripped to audio-only:

- Manifest: `utt_id<TAB>wav_path` (no text/speaker-id columns — codec
  training needs neither, and dropping them keeps the manifest usable
  directly as the `UnsortedBatchSampler` key file).
- `__getitem__` returns `{"audio": np.float32 array}` — key must be
  `"audio"` to match `CommonPreprocessor(speech_name="audio")`.
- `__getitem__` accepts both a positional int (plain `torch.utils.data.
  DataLoader` usage) and a utt_id string (`ESPnet`'s chunk/sequence
  iterators sample batches of utt_id keys via `UnsortedBatchSampler` and
  index datasets with those keys directly — found by an end-to-end smoke
  test, not obvious from a quick read of the reference TTS builder this was
  ported from, which only supports int indexing).
- Subsets: `train-clean-100 + train-clean-360 + train-other-500` /
  `dev-clean` / `test-clean` (`dataset/config.yaml`), same split as the
  reference TTS recipe.

## Model config

`conf/training_encodec.yaml` sets `task:
espnet2.tasks.gan_codec.GANCodecTask` and a `model:` block shaped like
`GANCodecTask`'s argparse namespace (`codec: encodec`, `codec_conf: {...}`),
consumed via `espnet3.utils.task_utils.get_espnet_model` (same pattern the
reference TTS recipe uses for `GANTTSTask` — `model:` is NOT a Hydra
`_target_` block in this path, it mirrors the task's CLI args directly).
`codec_conf` values are copied from
`egs2/libritts/codec1/conf/tuning/train_encodec.yaml`, the existing
ESPnet2 recipe for the same dataset/model.

Optimizers use the named multi-optimizer path (`optimizers.generator` /
`optimizers.discriminator`, matched by `params: generator` /
`params: discriminator` against `model.codec.generator.*` /
`model.codec.discriminator.*`, which `ESPnetGANCodecModel.__init__` asserts
exist).

## Worked around: hyphenated utt_ids (CombinedDataset int() trap)

Found via an end-to-end smoke test (build the dataset, then feed it through
the real `ChunkIterFactory`/`CombinedDataset` path exactly as production
training would).

`CombinedDataset.__getitem__` (`espnet3/components/data/dataset.py`,
**shared framework code, not touched here**) receives string utterance-ID
keys from `UnsortedBatchSampler`/`SequenceIterFactory` (used by the chunk
iterator) and tries `int(idx)` to decide whether to treat the key as a
positional index or fall through to utterance-ID lookup. Python's `int()`
accepts PEP 515 underscore-grouped numeric literals, so LibriTTS's native
utt_id convention (`speaker_chapter_utt_segment`, all-digit segments joined
by `_`, e.g. `"1089_134691_000004_000001"`) silently parses as the integer
`1089134691000004000001` instead of raising and falling through to the
utterance-ID path — throwing `IndexError: Index out of range in
CombinedDataset` on close to the first training batch.

This would affect any ESPnet3 recipe using LibriTTS-style utt_ids with the
chunk/sequence iterator path, not just this one, and the root cause is a
framework bug worth fixing there too (a strict digits-only check, e.g.
`idx.lstrip("-").isdigit()`, instead of the loose `int(idx)` try/except —
left as a separate, not-yet-filed follow-up against
`espnet3/components/data/dataset.py`).

For this recipe specifically, no framework change was needed:
`LibriTTSCodecBuilder._scan_subset_entries` (`dataset/builder.py`)
hyphenates utt_ids instead of using LibriTTS's native underscores (matching
LibriSpeech's own ID convention), which guarantees `int(idx)` raises and
`CombinedDataset` always takes the intended utterance-ID lookup path.
Verified end-to-end (build → dataset → `CombinedDataset` → `ChunkIterFactory`).

## Inference and metrics (deferred)

Not implemented. Codec reconstruction evaluation (encode -> decode
roundtrip, then PESQ/STOI/mel-cepstral distortion or similar) has no
existing ESPnet3 port to reuse, unlike ASR's WER/CER
(`espnet3/systems/asr/metrics/`). `conf/inference.yaml` / `conf/metrics.yaml`
are placeholders only (so `run.py` loads without error) — do not run
`--stages infer` or `--stages measure` yet. When picked up, port
`espnet2/bin/gan_codec_inference.py`'s encode/decode roundtrip into an
ESPnet3 provider/runner (see `egs3/mini_an4/asr/conf/inference.yaml` for the
provider/`output_fn` pattern).

## Reference material

- `espnet2/tasks/gan_codec.py` — ported task (class choices, model/optimizer
  wiring, preprocessor, required data names).
- `espnet2/gan_codec/espnet_model.py` — `ESPnetGANCodecModel`, confirms
  `audio` input key and empty `collect_feats`.
- `espnet2/gan_codec/encodec/encodec.py`,
  `egs2/libritts/codec1/conf/tuning/train_encodec.yaml` — Encodec
  generator/discriminator/loss defaults and the actual espnet2 recipe values
  carried into `conf/training_encodec.yaml`.
- `espnet2/train/abs_gan_espnet_model.py` — `AbsGANESPnetModel` contract
  (`forward_generator`, `optim_idx`).
- `espnet2/iterators/chunk_iter_factory.py`,
  `espnet2/samplers/build_batch_sampler.py`,
  `espnet2/tasks/abs_task.py` (`build_chunk_iter_factory`) — confirms chunk
  iteration's per-sample sampler is always `UnsortedBatchSampler(batch_size=
  1, key_file=...)`, independent of `batch_type`.
- `NewGamezzz/espnet:libritts_vits`, `egs3/libritts/tts/` — reference
  ESPnet3 TTS recipe this GAN adapter pattern and dataset builder pattern
  were ported from.
- `espnet3/components/modeling/lightning_module.py`,
  `espnet3/components/trainers/trainer.py` — base classes the shared GAN
  adapter (`espnet3/systems/codec/gan_lightning_module.py`,
  `espnet3/systems/codec/gan_trainer.py`) extends.

## Open items

- Exact optimizer LR/betas/schedule and `chunk_length` are carried as-is
  from `egs2/libritts/codec1/conf/tuning/train_encodec.yaml`; revisit once
  training runs are underway.
- `infer` / `measure` (see above).
