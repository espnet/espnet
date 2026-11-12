# ESPnet3 TSE Contract Migration Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This document authorizes no implementation, commits, pushes, or uploads.

**Goal:** Make PR #6357 a complete, clean-checkout ESPnet3 TSE system using the merged shared contracts, while preserving its training and scoring behavior.

**Architecture:** Use `ESP2TSESystem` and the shared launcher for orchestration. Keep corpus preparation in the LibriMix recipe, reusable adaptation in `systems/esp2_tse/src/`, and scoring in `metrics/`. Use the checked runner/provider boundary for both recipe and published inference; references come from the evaluation dataset.

**Tech Stack:** Python, ESPnet2/3, PyTorch/Lightning, Hydra/OmegaConf, soundfile, LibriMix, existing metric backends.

**Spec:** [Maintainer migration request](https://github.com/espnet/espnet/pull/6357#issuecomment-6329879802), [merged #6931](https://github.com/espnet/espnet/pull/6931), repository `AGENTS.md`, `CONTRIBUTING.md`, and `.agents/` package guides. The requested behavior and acceptance criteria are recorded below; remote example snippets are illustrative and must be reconciled with source.

## Global constraints

- Preserve existing working-tree changes, untracked tests, ignored dataset sources, and the older plan. Do not reset, stash indiscriminately, or clean this checkout.
- Keep the agreed `esp2_tse` name and class `ESP2TSESystem`. Root implementation files are `system.py`, the corresponding `task.py`, package initializers, `metrics/`, and `src/`.
- Use absolute imports in changed Python modules. Update every affected import, Hydra target, config overlay, resource include, test, and documentation path together.
- Preserve recipe `num_spk: 2`, enrollment placeholder lookup and `enroll_segment: 48000` cropping, target selection, checkpoint arguments, model architecture, optimizer, and scoring definitions. Do not substitute a one-speaker training model.
- Keep single-item inference (`batch_size: null`) until native batching is implemented and verified. WAVs carry the actual rate; the LibriMix recipe is 16000 Hz, whereas native `SeparateSpeech.__call__` defaults to 8000 Hz.
- Follow ESPnet3 Python recipe/config conventions; do not apply ESPnet2 shell recipe rules to ESPnet3. Do not reorganize unrelated upstream systems.
- Each new/changed test set covers valid behavior and an actionable failure. Use 30-second per-test timeouts per `.agents/ci/AGENTS.md`; this is more specific than CONTRIBUTING's generic 10-second guidance.
- Run integration scripts only after reading them and only in a disposable checkout with disposable output directories. No Hub upload is part of validation.

## Review focus

1. A fresh checkout must contain the dataset implementation, configs, and importable exports (Task 2).
2. Raw training enrollment placeholders must pass dataset checks and retain preprocessing/cropping; inference must receive actual enrollment audio (Tasks 2–3).
3. Unequal audio lengths, wrong rates/channels, missing enrollment, NaN/Inf, and unsupported batches must fail at the appropriate boundary (Tasks 3–4).
4. The first native extraction target, utterance IDs, WAV rate/layout, and dataset scoring references must agree across recipe and packed/demo paths (Tasks 3, 5–6).
5. Chunked metric updates, reset/state merge, error/NaN policy, and all configured score names must preserve corpus results (Task 4).

## Audit evidence and scope

Inspected 2026-11-06. Local HEAD and GitHub PR head both equal `0e0dd1f7712d07bdf291badd16af08788415cec2`. Local upstream reference/base is `a7d6268adf109f9eeca8afc0951a45d911e36933`; the PR diff contains 45 files, 3446 insertions and 17 deletions. Live master is `1cb08d02464dec1707bf35f460bd14ad356549d3`. The migration merge `9b18b5ce301df1571f7d3401f7bc9fce90266721` is absent locally. Read its launcher, dataset base, runner and metric base via GitHub; launcher, runner and metric base are unchanged on live master. A full current-master compatibility review remains an implementation prerequisite.

Read all repository AGENTS.md files: root review guidance, shared contributor guide, ESPnet3 index and components/systems/inference/publication/utils/parallel guides, recipe guide, and CI guide. The final parallel guide is needed before implementation if worker behavior is changed. Guides describe newer APIs absent from this branch; do not invent local substitutes for them.

| Priority | Evidence | Consequence / planned action |
| --- | --- | --- |
| P1 | `.gitignore:56` ignores `egs*/*/*/data*`; `git ls-files egs3/librimix/esp2_tse/dataset` returns nothing | Four local dataset source/config files exist but will not ship. Track source explicitly, with a narrow ignore exception. |
| P1 | `egs3/librimix/esp2_tse/run.py:7` uses `BaseSystem`; `espnet3/systems/esp2_tse/` has no `system.py` | Migrate to the required concrete System and shared launcher after bringing in upstream contracts. |
| P1 | `egs3/librimix/esp2_tse/conf/inference.yaml:61` onward targets raw SeparateSpeech and passes reference audio for output formatting | Replace legacy formatting with checked model adaptation; publish only `inf` and bind references from the dataset. |
| P1 | `espnet3/systems/esp2_tse/metrics/sisnr.py:35` constructor and analogous metric modules have no new state lifecycle | These classes must implement the merged BaseMetric declarations, initialization, update and compute contracts before using newer master. They remain valid against the old local BaseMetric. |
| P1 | Ignored `dataset/dataset.py:251–256` raises for any retained non-audio field, including `utt_id` | Default filtering conceals the issue; keeping IDs for evaluation fails. Return declared metadata only in the appropriate mode and exclude it from training features. |
| P2 | `espnet2/iterators/chunk_iter_factory.py:20–53` adds a marker and process-wide typeguard checker for one ListConfig argument | Normalize Hydra containers at the ESPnet3 construction boundary instead, with compatibility tests. |
| P2 | Untracked tests and CI script still reference `tse/`, `train.yaml`, `infer.yaml`, and legacy output_fn behavior | Port useful assertions to the current interfaces; do not mechanically rename obsolete expectations. |
| P2 | `egs3/librimix/esp2_tse/readme.md` and TEMPLATE lowercase README, copied runner/UI and reusable transform | Rename README files, delete superseded launcher/output helpers, move the transform into system src, and prefer the existing shared demo UI where compatible. |

The older `2026-05-22-librimix-tse-recipe.md` is historical and describes obsolete paths/APIs. Do not execute it alongside this plan. Existing modified `components/data/__init__.py` and `test_epoch_sync_iterator.py` are separate local work: inspect and preserve them; include only justified changes in final PR scope.

## Task 1: Establish the migration baseline and launcher

**Files:** Add `espnet3/systems/esp2_tse/system.py`; modify recipe `run.py` and TEMPLATE role configs; delete `egs3/TEMPLATE/esp2_tse/run.py`; migrate runner tests into `test/espnet3/systems/esp2_tse/test_system.py` and recipe config tests under `test/egs3/librimix/esp2_tse/`.

**Interfaces:** `ESP2TSESystem(BaseSystem)` exposes supported shared stages and their config roles; recipe entry calls `launch(ESP2TSESystem)`. Use actual upstream StageSpec/launcher signatures, not the older test suite's invented dataset_dir requirement.

- [ ] Preserve a file inventory including ignored dataset code before any merge/rebase; prepare a separate disposable implementation checkout if needed without losing those sources.
- [ ] Bring current upstream into the implementation branch and resolve conflicts with the TSE delta. Re-read `.agents/`, actual contract/data/metrics/stages code and current TEMPLATEs. Recompute the PR diff; do not replay inherited framework changes as new TSE edits.
- [ ] Write failing launcher/config tests: class/default package resolution, declared stage order, CLI help, supported config roles, and a missing required config producing an actionable error. Run the focused tests to confirm the intended failures.
- [ ] Add the minimal System, switch the recipe launcher and remove superseded TEMPLATE runner. Keep default System stages; add overrides only for real TSE behavior.
- [ ] Resolve every role's defaults and recipe overlays, including tuning, publication and demo. Tests must import real targets and cover invalid overrides. Re-run the focused tests, then commit this coherent migration slice.

## Task 2: Ship a contract-correct LibriMix dataset and builder

**Files:** `.gitignore`; `egs3/librimix/esp2_tse/dataset/{__init__.py,builder.py,dataset.py,config.yaml}`; training/inference config overlays; new `test/egs3/librimix/esp2_tse/test_dataset.py` and `test_builder.py`.

**Interfaces:** Export exact `Dataset`/`DatasetBuilder` names. Dataset inherits `BaseDataset`, returns declared raw fields, and accepts explicit `training_placeholders: bool = False` in addition to existing split/path/filter arguments. Builder retains source preparation versus task artifact preparation lifecycle.

- [ ] Add fixtures containing a few tiny WAVs, manifests and enrollment maps. Assert train placeholders survive raw contract checks; dev/test enrollment resolves to audio paths; missing files, unknown splits, duplicate/mismatched manifest IDs and missing enrollment fail clearly. Cover retained `utt_id` and metadata filtering.
- [ ] Confirm tests fail for current behavior, then adapt the existing ignored sources rather than replacing their corpus logic with a skeleton. Preserve 2-/3-speaker fields where supported; declarations must match each instance's returned fields and enrollment mode.
- [ ] Make builder checks cheap and side-effect free. Move manifest/cache/enrollment-map preparation into build, with raw downloads/source checks in prepare_source; test checks do not download/write. Avoid silently simulating a corpus during dataset iteration/loading.
- [ ] Keep corpus URLs, roots, splits and run defaults config-wired; remove import-time config globals where instance/config loading suffices. Respect explicit external data_dir without requiring local LibriSpeech.
- [ ] Add a narrow source-only exception for this recipe's dataset Python/YAML files. Verify `git check-ignore` and staged `git ls-files` include all four source files while generated data remains ignored. Run dataset/builder tests and a fresh-checkout source/import check before committing.

## Task 3: Adapt native TSE inference at the checked boundary

**Files:** Add `espnet3/systems/esp2_tse/src/{__init__.py,inference_runner.py}`; move recipe `src/transforms.py` to system `src/transforms.py`; update all TEMPLATE/recipe training and inference YAML; remove both legacy recipe/TEMPLATE `src/inference.py`; tests in system `src/` and recipe test areas.

**Interfaces:** Prefer `ESP2TSEInferenceRunner(InferenceRunner)` with inherited `forward`, `prepare_model_inputs(inputs)` and `format_output(result, *, batched)`. Model remains native SeparateSpeech unless actual current code requires a backend adapter. Do not implement both routes or a redundant wrapper.

- [ ] Test checked runner calls with a recording native-backend double: declared mixture/enrollment audio becomes native arrays, `fs=16000` reaches the backend, output equals the old first-target selection, and runner retains IDs. Cover genuine native `(T,)`/`(1,T)` output layouts and unsupported batch failure.
- [ ] Test invalid audio values, mismatched length where pairing requires it, missing inputs, wrong channels/rates under the selected policy, and NaN/Inf. Preserve all enrollment inputs actually required by supported configurations; the default evaluation selects target 1, not a changed training speaker count.
- [ ] Implement hooks only for format conversion. Pass the recipe rate via `provider.params.model_kwargs.fs`; keep `batch_size: null`. Retain rate information until validated and return `inf` as `Audio` at the model rate. If model-dependent invocation needs an adapter, put it under system `src/` and use the shared runner rather than duplicating checked forward logic.
- [ ] Put shared input/contract/runner/output defaults in TEMPLATE. Drop scoring references from model inputs/outputs and remove output_fn everywhere. Move the reusable audio transform and update training/tuning targets while preserving enrollment cropping inside TSEPreprocessor.
- [ ] Run focused runner/transform tests and write/read real tiny WAV artifacts to confirm samples, layout, IDs and 16000-Hz headers. Commit after native first-target parity passes.

## Task 4: Migrate metrics without changing score definitions

**Files:** All six `espnet3/systems/esp2_tse/metrics/{tsos,pesq,stoi,sisnr,sdr,dnsmos}.py`; TEMPLATE and recipe `conf/metrics.yaml`; focused tests under `test/espnet3/systems/esp2_tse/metrics/`.

**Interfaces:** Each metric calls `super().__init__()`, declares typed inputs/number outputs, implements `update(**values: Sequence[Any]) -> None` and `compute() -> Dict[str, float]`, and uses `add_state`/reset/merge according to merged BaseMetric. Exact outputs: TSOS; PESQ; STOI or ESTOI; SI_SNR; SDR/SAR/SIR; DNSMOS_OVRL/DNSMOS_SIG/DNSMOS_BAK/DNSMOS_P808_MOS.

- [ ] Add score-parity fixtures and deterministic scorer doubles. Verify one update versus multiple chunks, reset, merged state, empty input policy, and metric-specific error/NaN handling. Keep real backend checks separate from optional-dependency skips.
- [ ] Resolve raw SCP path versus Audio typing before implementing update: merged BaseMetric checks values before the update body and does not decode SCP strings. Use the existing path-accepting audio Kind/loaders if supported; otherwise a system-owned, registered path-aware Kind with loading/roundtrip tests. Do not declare audio then decode too late inside update.
- [ ] Replace each custom full-dataset __call__/line-count cache with the shared lifecycle, scalar sums/counts and appropriate artifact hooks. Keep numerical formulas, target/channel policies, score scaling and backend options. Test shape validation after channel selection and padding masks; collate helpers are private.
- [ ] Configure per-entry `inputs: {ref: dataset:speech_ref1, hyp: inf}` using block YAML, DNSMOS hyp only, and dataset_artifacts.speech_ref1 WAV materialization at 16000 Hz. Remove ref_key/hyp_key only with the constructor migration. Recipe metric lists replace the whole inherited list: retain every intended metric and mapping.
- [ ] Test missing/misaligned IDs, invalid audio/path/rate and unequal lengths; confirm reference artifacts are WAV, not .npy. Run all focused metric/config tests and commit.

## Task 5: Preserve ESPnet2 compatibility and simplify construction

**Files:** `espnet2/bin/{enh_inference,enh_tse_inference}.py`, `espnet2/enh/espnet_model.py`, `espnet2/train/preprocessor.py`, `espnet2/iterators/chunk_iter_factory.py`, ESPnet3 `components/data/dataloader.py` if needed, and matching `test/espnet2/` / `test/espnet3/components/data/` tests.

**Interfaces:** Preserve pre-existing batched ESPnet2 inference and preprocessor behavior. Convert OmegaConf containers into ordinary Python containers at the ESPnet3 factory-construction boundary, retaining ChunkIterFactory's native public types and validation.

- [ ] Add regressions for unbatched and batched mixture/enrollment inference; string versus ndarray enrollment; intermediate-output enhancement normalization/resampling; and chunk iterator enrollment exclusions. Run them first to establish which fixes remain necessary after the upstream update.
- [ ] Test ordinary iterator factory config and an OmegaConf excluded_key_prefixes list through actual construction and iteration, including invalid element rejection. Set the narrow Hydra conversion policy there; remove this PR's global typeguard marker/registration when covered.
- [ ] Preserve the existing epoch-sync local work; avoid adding library API solely to simplify tests. Run relevant ESPnet2 and data component tests before committing. Consider a separate compatibility PR if maintainers want the independent enhancement-model fix split out.

## Task 6: Verify packaging, demos, documentation and real CI coverage

**Files:** TEMPLATE/recipe publication/demo configs and resource includes; README files and model/demo card templates; new `egs3/mini_an4/esp2_tse/` fixture; repair local `ci/test_integration_espnet3_tse.sh` or extend existing integration script; relevant workflow/relevance selection; `test/ci/` and publication tests.

**Interfaces:** The same checked runner/contract serves recipe infer, `InferenceModel.from_packed` and DemoSession. References are evaluation inputs, never required demo/model outputs. Packed paths and custom code load outside the source recipe.

- [ ] Test tiny-model pack/load from outside the recipe and single-item prediction, including correct rate, enrollment binding and `inf` output. Test missing/untrusted user code and invalid bundle paths. Exercise the actual shared demo callback and audio output conversion.
- [ ] Audit whether the existing shared Gradio launcher meets this two-audio-input/one-audio-output UI; reuse it if so. Retain custom code only for actual UI behavior. Update all moved-code resources, trust settings and contract imports; remove stale forward_fs settings when superseded by the shared rate policy.
- [ ] Rename lowercase readme.md files to README.md. Document source preparation and copy-pasteable stage commands against actual config paths, plus optional scoring dependencies/checkpoints. Preserve reported results but do not claim they were reproduced.
- [ ] Replace the stale integration script's nonexistent paths, fallback install and AST-only check with a disposable mini_an4-based TSE flow through supported create_dataset/collect_stats/train/infer/measure/pack stages. Use `coverage run --append` for Python execution. Connect it to a real CI job/flow and relevant change-path selection; a standalone script is insufficient.
- [ ] Run focused tests with the 30-second timeout options from the CI guide, relevant shared contract/launcher/publication tests, changed-file black/isort/pycodestyle and `bash ci/test_flake8.sh espnet3`. Run broader unit/integration checks as scope warrants; serial/worker equivalence is required if worker behavior changes. Record unavailable corpus/checkpoint/optional-dependency checks explicitly.
- [ ] Inspect final diff and fresh checkout source completeness; keep unrelated existing local work out of the PR. Commit independently reviewable slices. Do not push or upload as part of this planning task.

## Checks actually performed during planning

- Git status, refs, tracked/ignored source inventory, PR diff, all relevant guidance, configs and call paths inspected. Read live PR/maintainer discussion and merged framework sources; verified selected current-master interfaces as noted above.
- `pytest -q --collect-only test/espnet3/systems/tse/ test/egs3/TEMPLATE/tse/ test/egs3/librimix/tse/`: 42 tests collected, one collection error (exit 2). System import fails through Lightning/torchmetrics/ONNX Runtime; a NumPy ABI `_ARRAY_API not found` diagnostic is present. These are old local untracked tests, not current upstream acceptance tests.
- `pytest -o addopts='' -q --collect-only test/espnet3/systems/tse/test_system.py --tb=short`: repeated collection failure through the same dependency chain. No System behavior verified.
- `pytest -o addopts='' -q test/egs3/librimix/tse/test_transforms.py`: eight setup errors because tests load nonexistent `egs3/librimix/tse/src/transforms.py`.
- No implementation, framework merge, training, corpus download, integration run, model packaging or upload performed. This plan is the only intentional repository source addition.

## Recommended sequencing

Implement Tasks 1–4 as the required migration, then compatibility cleanup and end-to-end validation (Tasks 5–6). Prefer native execution in this session after plan review because these tasks share config and inference interfaces. Keep separate commits; splitting into separate PRs is optional and must leave each PR usable. A large rewrite of task/model architecture is unnecessary.
