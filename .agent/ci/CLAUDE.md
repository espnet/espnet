# `ci/` -- CI scripts relevant to espnet3

See [`.agent/CLAUDE.md`](../CLAUDE.md) for cross-cutting guidance (dev setup and
docstring/naming conventions). `ci/` is flat -- no subdirectories.

```
ci/
├── install.sh, install_kaldi.sh, install_macos.sh    # base / optional / platform dependency installers
├── check_kaldi_symlinks.sh, check_image_hash_inputs.py  # CI image / repo hygiene checks
├── licence_audited.txt, no_redistribute.txt, report_unlicensed.py  # license bookkeeping
├── doc.sh                                # builds the Sphinx + VuePress documentation site
├── test_flake8.sh                        # flake8 (+ flake8-docstrings for a package allowlist) wrapper
├── test_import_all.py                    # import-smoke-tests every module in the repo
├── test_utils.sh
├── test_configuration_espnet2.sh, test_python_espnet2.sh,
│   test_shell_espnet2.sh, test_integration_espnet2.sh   # espnet2 CI legs (not espnet3)
├── test_python_espnet3.sh                # espnet3 unit-test CI leg (see below)
├── test_integration_espnet3.sh           # espnet3 integration CI leg (see below)
├── test_integration_espnet3_publication.sh  # publication CI leg (see below)
├── test_integration_espnet3_publication_check.py  # helper invoked by the script above
└── test_demo_ui.py                       # Playwright driver for the packed demo UI
```

## The three espnet3 CI legs

- **`test_python_espnet3.sh`** -- the unit-test leg: `test_flake8.sh espnet3` -> `pycodestyle` over
  `espnet3`/`test/espnet3`/`ci` -> `pytest -q --execution-timeout 10.0 test/espnet3/` -> coverage
  report. This is what section 5 of the root guide ("run locally before opening a PR") mirrors.
- **`test_integration_espnet3.sh`** -- clones `mini_an4/asr` via `espnet3 clone` (so the clone codepath
  itself is exercised, not just the in-place recipe), then runs
  `create_dataset -> train_tokenizer -> collect_stats -> train -> infer -> measure` for four training
  configs in a row (`training_asr_streaming.yaml`, `training_asr_transformer.yaml`,
  `training_asr_transducer.yaml`, and `training_transducer_asr_conformer_rnnt.yaml` with its own
  inference config), wiping `exp/`/`data/` between runs.
- **`test_integration_espnet3_publication.sh`** -- runs the same stage sequence through `pack_model`
  (optionally `upload_model` when `ESPNET3_PUBLICATION_TEST_UPLOAD=true` and a Hugging Face repo is
  configured), validates the packed bundle from *outside* the recipe tree with
  `test_integration_espnet3_publication_check.py` (using `espnet3.publication.InferenceModel`) to catch
  accidental relative-path dependencies, then runs `pack_demo` twice (default UI and a custom
  multi-input UI, both synthesized inline in the script) and drives the resulting Gradio app with
  Playwright via `test_demo_ui.py`.

If you add a new stage or change what `pack_model`/`pack_demo` produce, these two integration scripts
are what "full workflow" coverage means in this repo (root guide, section 5) -- extend them rather than
adding a fourth parallel script, unless the new stage genuinely does not fit either flow.

## `test_flake8.sh`

Wraps two independent checks: `flake8 --extend-ignore=D` over legacy directories (docstring rule `D`
disabled there), and a strict `flake8` (docstrings included) over whatever directory is passed as
`$1` -- currently `espnet3` (called as `test_flake8.sh espnet3` by `test_python_espnet3.sh`). New
`espnet3/` code is therefore always checked against `flake8-docstrings`; see the docstring guide in
the root `.agent/CLAUDE.md`.
