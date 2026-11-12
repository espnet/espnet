# ESPnet3 TSE migration validation

Implemented against upstream master `45e2b16117`, merged locally in `2b87bd5a0`.
No push, PR comment or model upload was performed.

The migration adds `ESP2TSESystem`, replaces the copied launcher, ships the previously
ignored LibriMix dataset/builder/config, separates training enrollment placeholders
from evaluation audio, and adapts native extraction through the shared checked runner.
All six metric modules now use typed inputs and mergeable update/reset/compute state.
References come from the dataset and are materialized as WAV rather than model outputs.
Native model/training/scorer settings and the first-target inference choice are retained.
Shared iterator construction now converts Hydra containers instead of registering a
global ESPnet2 type checker. Decoded enrollment supports the native cropping behavior.

The AN4 smoke fixture is connected to the ESPnet3 integration job through
`ci/test_integration_espnet3_tse.sh`. It clones into a temporary directory, uses
`coverage run --append`, and invokes `ci/check_espnet3_tse.py` from outside the recipe.

## Verification performed

- Final shared/TSE pytest run: **585 passed, 2 xfailed, 1 deselected**. Covers contracts,
  data components, launcher, publication, shared inference paths, TSE system/runner,
  seven metric classes, LibriMix manifests/configs/builders and the tiny AN4 dataset.
- The deselected test is the user's pre-existing local
  `test_single_rank_skips_the_collective`; it expects an optimization absent from the
  unchanged iterator implementation. It failed on the baseline and is left untouched.
- ESPnet2 enhancement/TSE inference, chunk iteration, decoded enrollment and relevant
  data-loader compatibility tests: **84 passed** in the earlier compatibility run.
  The newly added native unbatched-versus-single-batch TSE regression also passed.
  SVoice intermediate-output regression checks passed in all six training/eval and
  variance-normalization combinations.
- Fresh CLI-created clone completed `create_dataset collect_stats train infer measure
  pack_model pack_demo`. Actual tiny TD-SpeakerBeam training ran for one batch/epoch.
  Native backend, packed model, demo callback and generated WAV audio matched; output
  IDs, WAV sample rate and metrics artifact existence were checked from outside the recipe.
- Independent review found external enrollment-map routing, YAML noise-path conversion,
  writable noise-copy reuse, and incomplete `is_built` checks. Each was verified and fixed;
  the builder/config regressions pass.
- Black/isort check mode, scoped strict ESPnet3/recipe flake8, legacy lint scope,
  pycodestyle, shell syntax and `git diff --check` passed.

Logs and disposable experiment output are under `/tmp/espnet-tse-refactor-20261112/`.
The repository's generated `tools/activate_python.sh` is unavailable locally, so the
integration stages were run directly in an isolated Python 3.12 environment, rather
than claiming execution of the entire CI wrapper. Local Dask/Gloo tests needed socket
permission. The test environment uses installed Torch 2.9.1 and NumPy 2.2.6; it is not
an exact reproduction of the upstream CI image. ONNX Runtime/soxr were updated only
in that isolated environment to resolve NumPy ABI problems, and Numba used a writable
cache with JIT explicitly enabled for shared resampling tests.

## Remaining verification boundaries

Full LibriMix simulation/training and published benchmark scores were not reproduced.
PESQ/STOI/SDR/DNSMOS lifecycle tests use deterministic scorer doubles where appropriate;
real optional scorer models/dependencies and GPU execution were not comprehensively run.
No full-repository CI run or browser UI test is claimed. Existing unrelated local
AGENTS.md, lazy data exports, epoch-iterator test edits and scratch files are preserved.
Original local-only dataset/tests/guidance were backed up before the merge; obsolete
untracked tests are retained under the ignored execution workspace after useful
behavior assertions were ported to current paths.
