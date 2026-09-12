# Contributing to ESPnet

Thanks for taking the time to contribute. Issues, questions, and pull requests are all
welcome — open an [issue](https://github.com/espnet/espnet/issues) or join us on
[Discord](https://discord.gg/hrCs85gFWM) if you are not sure where to start.

> [!NOTE]
> Contributions target ESPnet2 (`espnet2/`, `egs2/`) or ESPnet3 (`espnet3/`, `egs3/`).
> ESPnet1 reached end of life and is no longer part of this repository.

## 1. What to contribute

Contributions usually fall into three categories: major features, minor updates, and recipes.

### 1.1 Major features

If you want to ask or propose a new feature, please first open a new issue with the tag
`Feature request`, or directly contact Shinji Watanabe <shinjiw@ieee.org> or other main
developers. Each feature implementation and design should be discussed and modified
according to ongoing and future works.

You can find ongoing major development plans at
[milestones](https://github.com/espnet/espnet/milestones) or in the pinned
[issues](https://github.com/espnet/espnet/issues).

### 1.2 Minor updates (minor feature, bug fix for an issue)

If you want to propose a minor feature, update an existing one, or fix a bug, please first
take a look at the existing [issues](https://github.com/espnet/espnet/issues) and
[pull requests](https://github.com/espnet/espnet/pulls). Pick an issue and comment on the
task you want to work on.

If you need help or additional information to propose the feature, you can open a new issue
with the tag `Discussion` and ask ESPnet members.

### 1.3 Recipes

ESPnet provides and maintains many example scripts, called `recipes`, demonstrating how to
use the toolkit. Each subdirectory of `egs2` and `egs3` corresponds to a corpus.

| | Recipes | Entry point | Configuration |
| :-- | :-- | :-- | :-- |
| **ESPnet2** | [`egs2/`](egs2) | `run.sh` (shared `asr.sh`, `tts.sh`, `enh.sh`, ...) | YAML under `conf/` |
| **ESPnet3** | [`egs3/`](egs3) | `run.py --stages ...` | OmegaConf / Hydra under `conf/` |

#### 1.3.1 ESPnet2 recipes

ESPnet2 applies a paradigm without dependencies on Kaldi's binaries, which makes it lighter
and more generalized. We do not recommend preparing the recipe's stages for each corpus;
use the common pipelines provided in `asr.sh`, `tts.sh`, and `enh.sh` instead. For details
on creating ESPnet2 recipes, please refer to
[egs2/TEMPLATE/README.md](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/README.md).

The common pipeline takes care of `RESULTS.md` generation, model packing, and uploading.

#### 1.3.2 ESPnet3 recipes

ESPnet3 recipes are Python entry points rather than shell stages. A recipe runs as:

``` console
$ python run.py --stages create_dataset --training_config conf/training.yaml
$ python run.py --stages train          --training_config conf/training.yaml
$ python run.py --stages infer          --inference_config conf/inference.yaml
$ python run.py --stages measure        --metrics_config conf/metrics.yaml
```

See [egs3/TEMPLATE](https://github.com/espnet/espnet/tree/master/egs3/TEMPLATE) for the
recipe layout, and [`egs3/mini_an4/asr`](egs3/mini_an4/asr) for the smallest working
example. Packing and uploading a trained model is part of the same interface:

``` console
$ python run.py --stages pack_model   --training_config conf/training.yaml --publication_config conf/publication.yaml
$ python run.py --stages upload_model --training_config conf/training.yaml --publication_config conf/publication.yaml
```

`upload_model` uses your local Hugging Face login, so run `hf auth login` first.

#### 1.3.3 Publishing models

ESPnet models are hosted on the [Hugging Face Hub](https://huggingface.co/espnet). You do
**not** need to be a member of the `espnet` organization to publish one — a model under your
own namespace is loaded by `from_pretrained` exactly like one under `espnet/`.

1. Create a Hugging Face account — https://huggingface.co/
2. Log in locally with `hf auth login` (older installations of `huggingface_hub` call this
   command `huggingface-cli login`). The token is under Settings > Access Tokens and needs
   write access.
3. Create the repository under your own namespace, from the Hub web UI or with
   `hf repos create <your-username>/<model-name>`.
4. Upload the contents of your recipe's `exp` directory. `hf upload` covers most cases; for
   a large or incremental upload, clone the repository outside the ESPnet tree, run
   `git lfs install`, and push. Check other models for similar tasks to confirm the
   directory structure.
5. Keep `espnet` in the model card's `tags:`, so the model is findable by tag search on the
   Hub alongside the rest of the ecosystem.
6. Link the model from your recipe's `RESULTS.md`.

Your own namespace is the normal home for a contributed model, and nothing about it is
second class — `from_pretrained("<your-username>/<model-name>")` behaves exactly like a
model under `espnet/`.

If a model should nevertheless live under `espnet/`, that does **not** require organization
membership. Hub pull requests work the same way GitHub's do, so:

1. Ask a maintainer to create an empty `espnet/<model-name>` repository.
2. Upload your files to it as a pull request — you need no write access for this:

   ``` console
   $ hf upload espnet/<model-name> <local_dir> --create-pr
   ```

   Without `--create-pr` the upload would try to write to `main` and fail. To add to a PR
   you already opened, pass `--revision refs/pr/<n>` instead.
3. A maintainer reviews and merges it, and the files land under `espnet/<model-name>`.

For ESPnet3, the same applies through `conf/publication.yaml`: set `upload_model.hf_repo` to
`<your-username>/<model-name>` instead of the default `espnet/...`.

Models published on Zenodo are legacy. To port one to the Hub, run
`./scripts/utils/upload_models_to_hub.sh "ZENODO_MODEL_NAME"` from `egs2/RECIPE/*`.

#### 1.3.4 Additional requirements for a new recipe

- Common/shared files and directories such as `utils`, `steps`, `asr.sh`, etc., should be
  linked using a symbolic link (e.g., `ln -s <source-path> <target-path>`). Please refer to
  existing recipes if you're unaware of which files/directories are shared. Note that in
  ESPnet2 some of them are generated automatically by
  [egs2/TEMPLATE/asr1/setup.sh](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/setup.sh).
- Default training and decoding configurations (i.e., the default one in `run.sh`) should
  be named respectively `train.yaml` and `decode.yaml` and put in `conf/`. Additional or
  variant configurations should be put in `conf/tuning/` and named according to their
  differences.
- If a recipe for a new corpus is proposed, add its name and information to
  [egs2/README.md](https://github.com/espnet/espnet/blob/master/egs2/README.md) and `db.sh`.

#### 1.3.5 Checklist before you submit a recipe PR

- [ ] be careful about the name of the recipe. It is recommended to follow the naming
      conventions of the other recipes
- [ ] common/shared files are linked with a **symbolic link** (see Section 1.3.4)
- [ ] cluster settings should be set as **default** (e.g., `cmd.sh`, `conf/slurm.conf`,
      `conf/queue.conf`, `conf/pbs.conf`)
- [ ] update `egs2/README.md` with the corresponding recipe
- [ ] add the corresponding entry for a new corpus to `egs2/TEMPLATE/asr1/db.sh` — that is
      the only real file, every other `db.sh` in the tree is a symbolic link to it
- [ ] try to **simplify** the model configurations. We recommend having only the best
      configuration for the start of a recipe. Please also follow the default rule defined
      in Section 1.3.4
- [ ] large meta-information (e.g., the keyword list) for a corpus should be maintained
      elsewhere than in the recipe itself
- [ ] results and pre-trained models are included with the recipe (recommended)
- [ ] the recipe runs from a clean checkout — no absolute paths, no files that only exist on
      your machine
- [ ] code style is clean: `pre-commit run --all-files`

`black` and `isort` are applied automatically by `pre-commit.ci`, so formatting alone will
not block your PR. Running `pre-commit` yourself just avoids the extra commit.

ESPnet3 recipes (`egs3`) have no `db.sh`, `cmd.sh` or shared shell stages; the equivalent
settings live in the recipe's `conf/*.yaml`. The rest of the checklist still applies.

## 2. Pull requests

When your feature or bug fix is ready, open a Pull Request at
https://github.com/espnet/espnet, or use the Pull Request button in your forked repository.
If you are not familiar with the process, see
[GitHub's guide to creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

**Keep pull requests small** — no more than 20 files and fewer than 1000 lines changed.
Large PRs may be rejected unless they involve necessary refactoring or reformatting. Split
your work into smaller PRs where you can.

Pull requests are reviewed automatically by CodeRabbit when they are opened. It does not
re-review on every push, so after pushing fixes ask for a fresh pass by commenting:

```
@coderabbitai full review
```

Use `full review`, not `review` — the latter only looks at commits that have not been
reviewed yet. See [.coderabbit.yaml](.coderabbit.yaml) for the settings in effect.

## 3. Version policy and development branches

1. After v0.10.6, we moved to year- and date-based version specifiers, e.g., `v.202204`
   means April 2022.
2. The version number is updated regularly (e.g., every two months or so) or when there are
   significant changes.

## 4. Unit testing

ESPnet's tests are located under `test/`. You can install the additional packages needed for
testing as follows:

``` console
$ cd <espnet_root>
$ . ./tools/activate_python.sh
$ pip install -e ".[test]"
```

`tools/activate_python.sh` is generated when you set up the tools environment
(`tools/setup_python.sh`, `setup_venv.sh`, `setup_miniforge.sh`, ... then `make`), so it is
not in a fresh checkout. If you installed ESPnet into an environment you manage yourself,
activate that one instead and skip the line.

Testing various units thoroughly requires several modules and tools. We suggest reviewing
[ci/install.sh](https://github.com/espnet/espnet/blob/master/ci/install.sh), as it includes
everything required for the CI check.

### 4.1 Python

You can run the test suite with [pytest](https://docs.pytest.org/en/latest/) and
[coverage](https://pytest-cov.readthedocs.io/en/latest/reporting.html) by

``` console
$ ./ci/test_python_espnet2.sh   # pycodestyle, pytest test/espnet2, coverage
$ ./ci/test_python_espnet3.sh   # flake8, pycodestyle, pytest test/espnet3, coverage
```

Each script also runs the style checks, so a green run locally means the same checks pass in
CI. To iterate on a single test, call `pytest` directly on it.

Some useful tips when using `pytest`:

- A new test file should be put under `test/` and named `test_xxx.py`. Each method in it
  should have the format `def test_yyy(...)`. Pytest will find and run them automatically.
- We recommend adding several small test files instead of grouping them in one big file
  (e.g., `test_e2e_xxx.py`). Technically, a test file should only cover methods from one
  file (e.g., `test_transformer_utils.py` to test `transformer_utils.py`).
- To monitor test coverage and avoid overlapping tests, use
  `pytest --cov-report term-missing <test_file|dir>` to highlight covered and missed lines.
  For more details, see [coverage-test](https://pytest-cov.readthedocs.io/en/latest/readme.html).
- Each test is limited to 10.0 seconds — the scripts run
  `pytest --execution-timeout 10.0` (see
  [pytest-timeouts](https://pypi.org/project/pytest-timeouts/)). Use small model parameters
  and avoid dynamic imports, file access, and unnecessary loops. If a unit test genuinely
  needs more time, annotate it with `@pytest.mark.execution_timeout(sec)`, which overrides
  the default for that test.
- For test initialization (parameters, modules, etc.), use
  [pytest fixtures](https://docs.pytest.org/en/latest/fixture.html#using-fixtures-from-classes-modules-or-projects).

Please follow [PEP 8](https://peps.python.org/pep-0008/) for coding style and
[Google's convention](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
for docstrings.

### 4.2 Bash scripts

You can test the scripts in `utils` with [bats-core](https://github.com/bats-core/bats-core)
and [shellcheck](https://github.com/koalaman/shellcheck):

``` console
$ ./ci/test_shell_espnet2.sh
```

## 5. Integration testing

Write new integration tests in
[ci/test_integration_espnet2.sh](ci/test_integration_espnet2.sh) or
[ci/test_integration_espnet3.sh](ci/test_integration_espnet3.sh) when you add new features
in [espnet2/bin](espnet2/bin) or [espnet3](espnet3), respectively. They use our smallest
dataset, [egs2/mini_an4](egs2/mini_an4) and [egs3/mini_an4](egs3/mini_an4), to test a full
recipe run.

**Don't call `python` directly in integration tests. Use `coverage run --append`** as the
Python interpreter instead, so that the recipe run counts towards coverage.

In ESPnet2 the interpreter is passed into the shell pipeline, so `run.sh` must support
`--python ${python}`:

```bash
# ci/test_integration_espnet2.sh

python="coverage run --append"

cd egs2/mini_an4/your_task
./run.sh --python "${python}"
```

In ESPnet3 the recipe *is* a Python entry point, so the interpreter is used directly:

```bash
# ci/test_integration_espnet3.sh

python="coverage run --append"

cd egs3/mini_an4/asr
${python} run.py \
    --stages create_dataset train_tokenizer collect_stats train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

### 5.1 Configuration files

- [setup.cfg](setup.cfg) configures pytest, flake8, and isort.
- [.github/workflows](.github/workflows/) configures GitHub Actions (unit tests,
  integration tests).
- [codecov.yml](codecov.yml) configures CodeCov (code coverage).

### 5.2 Reproducing CI locally

CI jobs run inside a prebuilt container image (`docker/ci.dockerfile`), published to GHCR
and tagged with a hash of the files that determine its contents. The image is private, and
the workflow authenticates to GHCR with the job's `GITHUB_TOKEN`, so a local run cannot
simply pull it.

**Run the CI scripts directly.** This is what the jobs themselves execute, and it is enough
for most changes:

``` console
$ ./ci/test_python_espnet2.sh
$ ./ci/test_integration_espnet2.sh
```

**Reproduce the CI environment exactly.** Build the same image locally and run the script
inside it:

```bash
docker build -f docker/ci.dockerfile \
    --build-arg PYTHON_VERSION=3.12 --build-arg TH_VERSION=2.9.1 \
    -t espnet-ci:local .

docker run --rm -it -v "$PWD:/work" -w /work espnet-ci:local bash
```

Inside the container, wire your checkout to the baked-in environment the way
[.github/actions/use-prebuilt-environment](.github/actions/use-prebuilt-environment/action.yml)
does, then run the script:

```bash
for path in /espnet/tools/*; do
    name=$(basename "$path")
    [ -e "tools/${name}" ] || ln -s "$path" "tools/${name}"
done
pip install -e . --no-deps
pip install -r ci/no_redistribute.txt   # licences forbid baking these into the image
./ci/test_python_espnet2.sh
```

Those symlinks point inside the container. With a bind mount they are written into your
working tree and dangle once the container exits, so mount a throwaway clone rather than the
checkout you work in, or delete them afterwards.

The python and pytorch combinations that are actually built are listed in
[ci/image_variants.json](ci/image_variants.json); `python3 ci/image_variants.py pairs` prints
them.

> [!NOTE]
> [act](https://github.com/nektos/act) used to be the recommendation here. It cannot
> reproduce a full run any more: `resolve_ci_image` performs a `docker login ghcr.io` with
> the workflow's token and probes for published image tags, and the test jobs then run with
> `container.credentials` pointing at that same token. It remains usable for simple
> workflows that run directly on the runner.

## 6. Writing new tools

You can place your new tools under

- `espnet2/bin` or `espnet3`: heavy and large (e.g., neural network related) core tools.
- `utils`: lightweight, self-contained Python/Bash scripts.

For `utils` scripts, do not forget to add help messages and test scripts under `test_utils`.

### 6.1 Python tools guideline

Every module in `utils/` and `espnet2/bin/` should define `get_parser() -> ArgumentParser` at
module level. This is not a style preference: [ci/doc.sh](ci/doc.sh) runs
`doc/argparse2rst.py` over `./utils/*.py` and `./espnet2/bin/*.py`, which imports each file
and calls `get_parser()` to render its documentation page. A module without one raises
`ValueError: <path> does not have get_parser()`, which `argparse2rst.py` catches and logs —
the build keeps going and your tool silently ends up with no documentation. Give the parser
a `description` as well; it is the one-line summary in the tool index, and the renderer
asserts it is not `None`.

```python
#!/usr/bin/env python3
# Copyright XXX
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import argparse


# NOTE: argparse2rst.py imports this module and calls get_parser() to build
# the docs, so it must exist at module level and must not need arguments.
def get_parser():
    parser = argparse.ArgumentParser(
        description="awesome tool",  # DO NOT forget this
    )
    ...
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    ...
```

`espnet3/` is not part of that documentation pipeline, so the rule does not apply there.
Entry points under `espnet3/` are driven by recipe configs through `run.py` rather than by
command-line parsing.

### 6.2 Bash tools guideline

Scripts in `utils/` must support `--help`: [doc/usage2rst.sh](doc/usage2rst.sh) runs
`<script> --help` and captures the output as that tool's documentation page. If you use
Kaldi's `utils/parse_option.sh`, define `help_message="Usage: $0 ..."`.

> [!IMPORTANT]
> For ESPnet3, avoid adding shell scripts. ESPnet3 recipes are Python entry points
> (`run.py`) configured with OmegaConf / Hydra, and the stage logic that used to live in
> shell belongs in Python there. New ESPnet3 tooling should be Python unless there is a
> concrete reason it cannot be.

## 7. Writing documentation

See [doc/README.md](doc/README.md).

## 8. On CI failure

### 8.1 GitHub Actions

Open the failing check from the pull request's **Checks** tab and read the log of the step
that failed, or use the CLI:

``` console
$ gh run list --branch <your-branch>
$ gh run view <run-id> --log-failed     # only the failing steps
$ gh run rerun <run-id> --failed        # re-run just those jobs
```

<img width="725" alt="CI log location in the pull request checks tab" src="https://github.com/espnet/espnet/assets/11741550/e8e45c87-75e4-4489-a816-5c645b30fa0f">

A few behaviours are worth knowing before you conclude that your change broke something:

- **Draft pull requests skip the jobs.** They run once the PR is marked ready for review.
- **A new push cancels the run in progress** for that pull request. A cancelled job is not
  a failure.
- **If your PR touches the environment, the jobs get slower and fail differently.** The
  image tag is a hash of `ci/install.sh`, `ci/install_kaldi.sh`, `ci/no_redistribute.txt`,
  `docker/ci.dockerfile`, `pyproject.toml`, `tools/Makefile` and `tools/installers/**`.
  Changing any of them resolves to a tag that is only published after the PR merges, so the
  jobs build the environment from scratch instead of pulling it. The run says so with a
  notice, and a failure at that point may come from the build rather than from your change.
- **Network failures happen** — Hugging Face rate limits, package mirrors, GitHub Releases.
  Re-run the failed jobs before investigating.
- **Labels change what runs.** A PR labelled `Docker` without `ESPnet2` or `ESPnet3` runs
  only the docker path, and `test_upload` enables the publication upload test.

If you edited anything under `.github/` or `ci/`, run the invariant checker before pushing —
CI runs it first, and it fails the whole run:

``` console
$ python3 ci/check_ci_image_config.py
```

It enforces 14 rules that are invisible in review, among them: the file list that forms the
image hash is duplicated between `ci_on_ubuntu.yml` and `build_ci_image.yml` and must match
exactly; every python x pytorch combination a job requests must be one
`ci/image_variants.json` actually builds; every step running a `ci/test_*` script must have
`HF_TOKEN` in scope; and every third-party action must be pinned to a commit SHA.

### 8.2 Codecov

1. Write more tests to increase coverage.
2. Or explain to reviewers why you can't increase coverage.
