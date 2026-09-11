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

ESPnet models are maintained at [Hugging Face](https://huggingface.co/espnet). You can also
refer to the [ESPnet Model Zoo](https://github.com/espnet/espnet_model_zoo).

To upload a model manually:

1. Create a Hugging Face account — https://huggingface.co/
2. Request to be added to the espnet organization — https://huggingface.co/espnet
3. Log in with `hf auth login` (older installations of `huggingface_hub` call this
   command `huggingface-cli login`). The token is available under
   Settings > Access Tokens.
4. Create the model repository under the `espnet` organization, from the Hub web UI or
   with `hf repo create`.
5. `git clone https://huggingface.co/espnet/your-model-name` — clone this outside the
   ESPnet tree, since it is itself a git repository.
6. `cd your-model-name && git lfs install`
7. Copy the contents of your recipe's `exp` directory into it. Check other models for
   similar tasks under the espnet organization to confirm your directory structure.
8. `git add . && git commit -m "Add model files" && git push`
9. Check that the inference demo on the Hub runs successfully to verify the upload.

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
- [ ] add the corresponding entry in `egs2/TEMPLATE/*/db.sh` for a new corpus
- [ ] try to **simplify** the model configurations. We recommend having only the best
      configuration for the start of a recipe. Please also follow the default rule defined
      in Section 1.3.4
- [ ] large meta-information (e.g., the keyword list) for a corpus should be maintained
      elsewhere than in the recipe itself
- [ ] results and pre-trained models are included with the recipe (recommended)
- [ ] code style issues are resolved. You can run `utils/apply_code_fixes.py <folder>` to
      fix almost all of them automatically

We recommend the latest `black` and `isort` formatting. However, these are applied
automatically by `pre-commit.ci` and are no longer a requirement.

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

Testing various units thoroughly requires several modules and tools. We suggest reviewing
[ci/install.sh](https://github.com/espnet/espnet/blob/master/ci/install.sh), as it includes
everything required for the CI check.

### 4.1 Python

You can run the test suite with [pytest](https://docs.pytest.org/en/latest/) and
[coverage](https://pytest-cov.readthedocs.io/en/latest/reporting.html) by

``` console
$ ./ci/test_python_espnet2.sh
$ ./ci/test_python_espnet3.sh
```

Some useful tips when using `pytest`:

- A new test file should be put under `test/` and named `test_xxx.py`. Each method in it
  should have the format `def test_yyy(...)`. Pytest will find and run them automatically.
- We recommend adding several small test files instead of grouping them in one big file
  (e.g., `test_e2e_xxx.py`). Technically, a test file should only cover methods from one
  file (e.g., `test_transformer_utils.py` to test `transformer_utils.py`).
- To monitor test coverage and avoid overlapping tests, use
  `pytest --cov-report term-missing <test_file|dir>` to highlight covered and missed lines.
  For more details, see [coverage-test](https://pytest-cov.readthedocs.io/en/latest/readme.html).
- We limit each test to 2.0 seconds (see
  [pytest-timeouts](https://pypi.org/project/pytest-timeouts/)). Use small model parameters
  and avoid dynamic imports, file access, and unnecessary loops. If a unit test genuinely
  needs more time, annotate it with `@pytest.mark.execution_timeout(sec)`.
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
Python interpreter instead. In particular, `run.sh` should support `--python ${python}` so
it can call the custom interpreter.

```bash
# ci/test_integration_espnet2.sh

python="coverage run --append"

cd egs2/mini_an4/your_task
./run.sh --python "${python}"
```

### 5.1 Configuration files

- [setup.cfg](setup.cfg) configures pytest, flake8, and isort.
- [.github/workflows](.github/workflows/) configures GitHub Actions (unit tests,
  integration tests).
- [codecov.yml](codecov.yml) configures CodeCov (code coverage).

### 5.2 Running GitHub Actions locally

You can check whether your PR passes the integration tests before pushing, using
[act](https://github.com/nektos/act).

#### 5.2.1 Installation

1. Install [Docker](https://docs.docker.com/engine/install/) on your local machine. Do not
   forget to log in with `docker login`.
2. Install the GitHub CLI. The [instructions](https://github.com/cli/cli#installation)
   depend on your OS; for Linux you can use the
   [official sources](https://github.com/cli/cli/blob/trunk/docs/install_linux.md#official-sources).
3. Install **act** through a
   [package manager](https://github.com/nektos/act#installation-through-package-managers) or
   as a [GitHub CLI extension](https://github.com/nektos/act#installation-as-github-cli-extension).
   For Linux: `gh extension install https://github.com/nektos/gh-act`

#### 5.2.2 Usage

```bash
cd <root_dir_espnet_clone>  # go to the root directory of your clone
gh act
```

This runs all the CI tests that would run on the GitHub Actions server. For specific
jobs or workflows:

```bash
# For jobs:
gh act -j <jobID>  # Where jobID is a string.

# For workflows:
gh act -W <workflowID>
gh act -W .github/workflows/<filename>.yml
```

List the available job and workflow IDs with `gh act -l`. You can get the list of workflow
files from `ls .github/workflows`.

## 6. Writing new tools

You can place your new tools under

- `espnet2/bin` or `espnet3`: heavy and large (e.g., neural network related) core tools.
- `utils`: lightweight, self-contained Python/Bash scripts.

For `utils` scripts, do not forget to add help messages and test scripts under `test_utils`.

### 6.1 Python tools guideline

To generate a doc, do not forget `def get_parser(): -> ArgumentParser` in the main file.

```python
#!/usr/bin/env python3
# Copyright XXX
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import argparse

# NOTE: do not forget this
def get_parser():
    parser = argparse.ArgumentParser(
        description="awesome tool",  # DO NOT forget this
    )
    ...
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    ...
```

### 6.2 Bash tools guideline

To generate a doc, support `--help` to show its usage. If you use Kaldi's
`utils/parse_option.sh`, define `help_message="Usage: $0 ..."`.

## 7. Writing documentation

See [doc/README.md](doc/README.md).

## 8. On CI failure

### 8.1 GitHub Actions

Read the log from PR checks > details.

<img width="725" alt="CI log location in the pull request checks tab" src="https://github.com/espnet/espnet/assets/11741550/e8e45c87-75e4-4489-a816-5c645b30fa0f">

### 8.2 Codecov

1. Write more tests to increase coverage.
2. Or explain to reviewers why you can't increase coverage.
