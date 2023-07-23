# How to contribute to ESPnet

## 1. What to contribute
If you are interested in contributing to ESPnet, your contributions will fall into three categories: major features, minor updates, and recipes.

### 1.1 Major features

If you want to ask or propose a new feature, please first open a new issue with the tag `Feature request`
or directly contact Shinji Watanabe <shinjiw@ieee.org> or other main developers. Each feature implementation
and design should be discussed and modified according to ongoing and future works.
You can find ongoing major development plans at https://github.com/espnet/espnet/milestones
or at https://github.com/espnet/espnet/issues (pinned issues)

### 1.2 Minor Updates (minor feature, bug-fix for an issue)

If you want to propose a minor feature, update an existing minor feature, or fix a bug, please first take a look at
the existing [issues](https://github.com/espnet/espnet/pulls) and/or [pull requests](https://github.com/espnet/espnet/pulls).
Pick an issue and comment on the task that you want to work on this feature.

If you need help or additional information to propose the feature, you can open a new issue with the tag `Discussion` and ask ESPnet members.

### 1.3 Recipes

ESPnet provides and maintains many example scripts, called `recipes`, demonstrating how to
use the toolkit.  The recipes for ESPnet1 are put under `egs` directory, while ESPnet2 ones are put under `egs2`.
Like Kaldi, each subdirectory of `egs` and `egs2` corresponds to a corpus, which we have example scripts for.

#### 1.3.1 ESPnet1 recipes

ESPnet1 recipes (`egs/X`) follow the convention from [Kaldi](https://github.com/kaldi-asr/kaldi) and may rely on
several utilities available in Kaldi. As such, porting a new recipe from Kaldi to ESPnet is natural, and the user
may refer to [port-kaldi-recipe](https://github.com/espnet/espnet/wiki/How-to-port-the-Kaldi-recipe-to-the-ESPnet-recipe%3F)
and other existing recipes for new additions. For the Kaldi-style recipe architecture, please refer to
[Prepare-Kaldi-Style-Directory](https://kaldi-asr.org/doc/data_prep.html).

For each recipe, we ask you to report experimental results, environment, and model information.
For reproducibility, a link to upload the pre-trained model may also be added. All this information should be written
in a markdown file called `RESULTS.md` and put at the recipe root. You can refer to
[tedlium2-example](https://github.com/espnet/espnet/blob/master/egs/tedlium2/asr1/RESULTS.md) for an example.

To generate `RESULTS.md` for a recipe, please follow the following instructions:
- Execute `~/espnet/utils/show_result.sh` at the recipe root (where `run.sh` is located).
You'll get your environment information and evaluation results for each experiment in a markdown format.
From here, you can copy or redirect text output to `RESULTS.md`.
- Execute `~/espnet/utils/pack_model.sh` at the recipe root to generate a packed ESPnet model called `model.tar.gz`
and output model information. Executing the utility script without argument will give you the expected arguments.
- Put the model information in `RESULTS.md` and the model link if you're using a private web storage
- If you don't have private web storage, please contact Shinji Watanabe <shinjiw@ieee.org> to give you access to ESPnet storage.

#### 1.3.2 ESPnet2 recipes

ESPnet2's recipes correspond to `egs2`. ESPnet2 applies a new paradigm without dependencies on Kaldi's binaries, which makes it lighter and more generalized.
For ESPnet2, we do not recommend preparing the recipe's stages for each corpus but using the common pipelines, we provided in `asr.sh`, `tts.sh`, and
`enh.sh`. For details on creating ESPnet2 recipes, please refer to [egs2-readme](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/README.md).

The common pipeline of ESPnet2 recipes will take care of the `RESULTS.md` generation, model packing, and uploading. ESPnet2 models are maintained at Hugging Face and Zenodo (Deprecated).
You can also refer to the document at https://github.com/espnet/espnet_model_zoo
To upload your model, you need first (This is currently deprecated, uploading to Huggingface Hub is preferred) :
1. Sign up to Zenodo: https://zenodo.org/
2. Create access token: https://zenodo.org/account/settings/applications/tokens/new/
3. Set your environment: % export ACCESS_TOKEN="<your token>"

To port models from zenodo using Hugging Face hub,
1. Create a Hugging Face account - https://huggingface.co/
2. Request to be added to espnet organization - https://huggingface.co/espnet
3. Go to `egs2/RECIPE/*` and run `./scripts/utils/upload_models_to_hub.sh "ZENODO_MODEL_NAME"`

To upload models using Huggingface-cli conduct the following steps:
You can also refer to https://huggingface.co/docs/transformers/model_sharing
1. Create a Hugging Face account - https://huggingface.co/
2. Request to be added to espnet organization - https://huggingface.co/espnet
3. Run huggingface-cli login (You can get the token request at this step under setting > Access Tokens > espnet token
4. `huggingface-cli repo create your-model-name --organization espnet`
5. `git clone https://huggingface.co/username/your-model-name` (clone this outside ESPNet to avoid issues as this a git repo)
6. `cd your-model-name`
7. `git lfs install`
8. copy contents from `exp` directory of your recipe into this directory (Check other models of similar tasks under ESPNet to confirm your directory structure)
9. `git add . `
10. `git commit -m "Add model files"`
11. `git push`
12. Check if the inference demo on HF is running successfully to verify the upload

#### 1.3.3 Additional requirements for a new recipe

- Common/shared files and directories such as `utils`, `steps`, `asr.sh`, etc., should be linked using
a symbolic link (e.g., `ln -s <source-path> <target-path>`). Please refer to existing recipes if you're
unaware of which files/directories are shared. Noted that in espnet2, some of them are automatically generated by https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/setup.sh.
- Default training and decoding configurations (i.e., the default one in `run.sh`) should be named respectively `train.yaml`
and `decode.yaml` and put in `conf/`. Additional or variant configurations should be put in `conf/tuning/` and named accordingly
to their differences.
- If a recipe for a new corpus is proposed, you should add its name and information to:
https://github.com/espnet/espnet/blob/master/egs/README.md if it's an ESPnet1 recipe,
or https://github.com/espnet/espnet/blob/master/egs2/README.md + `db.sh` if it's an ESPnet2 recipe.

#### 1.3.4 Checklist before you submit the recipe-based PR

- [ ] be careful about the name of the recipe. It is recommended to follow the naming conventions of the other recipes
- [ ] common/shared files are linked with **symbolic link** (see Section 1.3.3)
- [ ] cluster settings should be set as **default** (e.g., `cmd.sh` `conf/slurm.conf` `conf/queue.conf` `conf/pbs.conf`)
- [ ] update `egs/README.md` or `egs2/README.md` with the corresponding recipes
- [ ] add the corresponding entry in `egs2/TEMPLATE/db.sh` for a new corpus
- [ ] try to **simplify** the model configurations. We recommend having only the best configuration for the start of a recipe. Please also follow the default rule defined in Section 1.3.3
- [ ] large meta-information (e.g., the keyword list) for a corpus should be maintained elsewhere other than in the recipe itself
- [ ] also recommend including results and pre-trained models with the recipe
- Note that we recommend the users to use the latest `black` and `isort` formattings. However, these formattings are automatically performed by `pre-commit.ci` and are no longer a requirement.

## 2 Pull Request
If your proposed feature or bugfix is ready, please open a Pull Request (PR) at https://github.com/espnet/espnet
or use the Pull Request button in your forked repo. If you're not familiar with the process, please refer to the following guides:

- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/

## 3 Version policy and development branches

1. After v10.6, we moved our version policy with the year and date-based specifiers, e.g., v.202204 means April 2022.

2. The version number will be updated when we regularly (e.g., every two months or so) or we have significant changes.

## 4 Unit testing

ESPnet's testing is located under `test/`.  You can install additional packages for testing as follows:
``` console
$ cd <espnet_root>
$ . ./tools/activate_python.sh
$ pip install -e ".[test]"
```
In order to thoroughly test various units, it is necessary to install several modules and tools. We suggest that you review the contents of [install.sh](https://github.com/espnet/espnet/blob/master/ci/install.sh), as it includes all the modules and libraries required for the CI check.

### 4.1 Python

Then, you can run the entire test suite using [pytest](https://docs.pytest.org/en/latest/) with [coverage](https://pytest-cov.readthedocs.io/en/latest/reporting.html) by
``` console
./ci/test_python_espnet1.sh
./ci/test_python_espnet2.sh
```
The followings are some useful tips when you are using `pytest`:
- New test file should be put under `test/` directory and named `test_xxx.py`. Each method in the test file should
have the format `def test_yyy(...)`.  [Pytest](https://docs.pytest.org/en/latest/) will automatically find and test them.
- We recommend adding several small test files instead of grouping them in one big file (e.g., `test_e2e_xxx.py`).
Technically, a test file should only cover methods from one file (e.g., `test_transformer_utils.py` to test `transformer_utils.py`).
- To monitor the test coverage and avoid the overlapping test, we recommend using  `pytest --cov-report term-missing <test_file|dir>`
to highlight covered and missed lines. For more details, please refer to [coverage-test](https://pytest-cov.readthedocs.io/en/latest/readme.html).
- We limit test running time to 2.0 seconds (see: [pytest-timeouts](https://pypi.org/project/pytest-timeouts/)) for each trial. Thus, we recommend using small model parameters and avoiding dynamic imports, file access, and unnecessary loops.
If a unit test needs more running time, you can annotate your test with `@pytest.mark.execution_timeout(sec)`.
- For test initialization (parameters, modules, etc.), you can use `pytest` fixtures. Refer to  [pytest fixtures](https://docs.pytest.org/en/latest/fixture.html#using-fixtures-from-classes-modules-or-projects) for more information.

In addition, please follow the [PEP 8 convention](https://peps.python.org/pep-0008/) for the coding style and [Google's convention for docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).

### 4.2 Bash scripts

You can also test the scripts in `utils` with [bats-core](https://github.com/bats-core/bats-core) and [shellcheck](https://github.com/koalaman/shellcheck).

To test:

``` console
./ci/test_shell_espnet1.sh
./ci/test_shell_espnet2.sh
```

## 5 Integration testing

Write new integration tests in [ci/test_integration_espnet1.sh](ci/test_integration_espnet1.sh) or [ci/test_integration_espnet2.sh](ci/test_integration_espnet2.sh) when you add new features in [espnet/bin](espnet/bin) or [espnet2/bin](espnet2/bin), respectively. They use our smallest dataset [egs/mini_an4](egs/mini_an4) or [egs2/mini_an4](egs/mini_an4) to test `run.sh`. **Don't call `python` directly in integration tests. Instead, use `coverage run --append`** as a python interpreter. Especially, `run.sh` should support `--python ${python}` to call the custom interpreter.

```bash
# ci/test_integration_espnet{1,2}.sh

python="coverage run --append"

cd egs/mini_an4/your_task
./run.sh --python "${python}"

```

### 5.1 Configuration files

- [setup.cfg](setup.cfg) configures pytest, black and flake8.
- [.github/workflows](.github/workflows/) configures Github Actions (unittests, integration tests).
- [codecov.yml](codecov.yml) configures CodeCov (code coverage).

### 5.2 Integration testing Locally (Github Actions locally)

You can test if your PR complies with the integrations test before pushing a new update using [act](https://github.com/nektos/act).

### 5.2.1 Installation

To execute **act**:

1. You first need to install [Docker](https://docs.docker.com/engine/install/) on your local PC. Do not forget to login into docker using `docker login`.

2. Install Github CLI. The [instructions](https://github.com/cli/cli#installation) will change depending on your OS. For Linux, you can use the official sources to install the corresponding [package](https://github.com/cli/cli/blob/trunk/docs/install_linux.md#official-sources).

3. Finally, install **act** through [package managers](https://github.com/nektos/act#installation-through-package-managers) or using the [Github CLI](https://github.com/nektos/act#installation-as-github-cli-extension). For Linux, you can use the command: `gh extension install https://github.com/nektos/gh-act`

### 5.2.2 Usage

In a command console:

```bash
cd <root_dir_espnet_clone>  # go to the root directory of your clone
gh act
```

The program will start running all the CI tests that will run in the GitHub Action server.

For specific jobs/workflow, you can use:

```bash
# For jobs:
gh act -j <jobID>  # Where jobID is a string.

# For workflows:
gh act -W <workflowID>
gh act -W .github/workflows/<filename>.yml
```

List the available jobID/workflowID with: `gh act -l`.
You can get the list of workflow files from `ls .github/workflows`.

## 6 Writing new tools

You can place your new tools under
- `espnet/bin` or `espnet2/bin`: heavy and large (e.g., neural network related) core tools.
- `utils`: lightweight, self-contained python/bash scripts.

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
        description="awsome tool",  # DO NOT forget this
    )
    ...
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    ...
```

### 6.2 Bash tools guideline

To generate a doc, support `--help` to show its usage. If you use Kaldi's `utils/parse_option.sh`, define `help_message="Usage: $0 ..."`.


## 7 Writing documentation

See [doc](doc/README.md).

## 8 On CI failure

### Github Actions

1. read the log from PR checks > details

<img width="725" alt="image" src="https://github.com/espnet/espnet/assets/11741550/e8e45c87-75e4-4489-a816-5c645b30fa0f">

### 8.2 Codecov

1. write more tests to increase coverage
2. explain to reviewers why you can't increase coverage
