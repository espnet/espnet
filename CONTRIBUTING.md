# How to contribute to ESPnet

## 1. What to contribute
If you are interested in contributing to ESPnet, your contributions will fall into three categories: major features, minor updates, and recipes. 

### 1.1 Major features

If you want to propose a new feature and implement it, please post about your intended feature at the issues, 
or you can contact Shinji Watanabe <shinjiw@ieee.org> or other main developers. 
We shall discuss the design and implementation.
Once we agree that the plan looks good, go ahead and implement it.
You can find ongoing major development plans at https://github.com/espnet/espnet/milestones

### 1.2 Minor Updates (minor feature, bug-fix for an issue)

If you want to implement a minor feature or bug-fix for an issue, please first take a look at 
the existing [issues](https://github.com/espnet/espnet/pulls) and/or [pull requests](https://github.com/espnet/espnet/pulls).
Pick an issue and comment on the task that you want to work on this feature.
If you need more context on a particular issue, please ask us, and then we shall provide more information.
   
We also welcome if you find some bugs during your actual use of ESPnet and make a PR to fix them.

### 1.3 Recipes

ESPnet provides and maintains a lot of reproducible examples similar to Kaldi (called `recipe`).
The recipe creation/update/bug-fix is one of our major development items, and we really encourage 
you to work on it.

ESPnet currently supports two versions of recipes: `egs` and `egs2` for espnet and espnet2, respectively.
Each of them follows a different structure.

#### 1.3.1 ESPnet1 recipes

ESPnet1's recipes correspond to `egs`.ESPnet1 follows convention from [Kaldi](https://github.com/kaldi-asr/kaldi), and it is also based on several
utilities from Kaldi. This feature naturally makes it very simple to port a Kaldi recipe to ESPnet. If you port a Kaldi recipe to ESPnet, please see
[port-kaldi-recipe](https://github.com/espnet/espnet/wiki/How-to-port-the-Kaldi-recipe-to-the-ESPnet-recipe%3F) for more detailed instructions.
If there is no existing Kaldi's recipe, you could still refer to 
[port-kaldi-recipe](https://github.com/espnet/espnet/wiki/How-to-port-the-Kaldi-recipe-to-the-ESPnet-recipe%3F) and the major task is to prepare
a Kaldi-style directory (Please see [Prepare-Kaldi-Style-Directory](https://kaldi-asr.org/doc/data_prep.html) for details.)
   
For each recipe, we also encourage you to report your results with your detailed environmental info and upload the model for the reproducibility 
(e.g., see [tedlium2-example](https://github.com/espnet/espnet/blob/master/egs/tedlium2/asr1/RESULTS.md)).
   
To make a report for `RESULTS.md`
 - execute `show_result.sh` at a recipe main directory (where `run.sh` is located), as follows. 
   You'll get environmental information and the evaluation result of each experiment in a markdown format.
   ```
   $ show_result.sh
   ```
 - execute `pack_model.sh` at the main directory of a recipe as follows. You'll get model information in a markdown format
   ```
   $ pack_model.sh --lm <language model> --dict <dict> <tr_conf> <dec_conf> <cmvn> <e2e>
   ```
 - `pack_model.sh` also produces a packed espnet model (`model.tar.gz`). If you upload this model to somewhere with a download link,
   please put the link information to `RESULTS.md`.
 - please contact Shinji Watanabe <shinjiw@ieee.org> if you want web storage to put your model files.

#### 1.3.2 ESPnet2 recipes

ESPnet2's recipes correspond to `egs2`. ESPnet2 applies a new paradigm without dependencies of Kaldi's binaries, which makes it lighter and more generalized.
For ESPnet2, we do not recommend preparing recipe's stages for each corpus but rather using the common pipelines we provided in `asr.sh`, `tts.sh`, and 
`enh.sh`. For details of creating ESPnet2 recipes, please refer to [egs2-readme](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/README.md).

The common pipeline of ESPnet2 recipes will take care of the `RESULTS.md` generation, model packing, and uploading. ESPnet2 models are maintained at Zenodo.
To upload your model, you need first:
1. Sign up to Zenodo: https://zenodo.org/
2. Create access token: https://zenodo.org/account/settings/applications/tokens/new/
3. Set your environment: % export ACCESS_TOKEN="<your token>"

#### 1.3.3 Additional Checklist when building recipes

- [ ] common files are linked with symlink: we ask to use symlink to refer common scripts and utilities (e.g., `utils`, `steps`, `asr.sh`). Please use `ln -sf <source_path> <target path>` rather than a new copy.
- [ ] configuration files are formated: major configuration file is named as `conf/train.yaml` and `conf/decode.yaml` while other options are kept in `conf/tuning`
- [ ] RESULT.md prepare: results are updated and prepared follow 1.3.1 or 1.3.2
- [ ] corpus registration: please register your target corpus in https://github.com/espnet/espnet/blob/master/egs2/README.md or https://github.com/espnet/espnet/blob/master/egs/README.md and `db.sh` (for ESPnet2 only)


## 2 Pull Request
Once you finish implementing a feature or bug-fix, please send a Pull Request to https://github.com/espnet/espnet

If you are not familiar with creating a Pull Request, here are some guides:

- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/

## 3 Version policy and development branches

We basically develop in the `master` branch.

1. We will keep the first version digit `0` until we have some super major changes in the project organization level.

2. The second version digit will be updated when we have major updates, including new functions and refactoring, and 
   their related bug fix and recipe changes.
   This version update will be done roughly every half year so far (but it depends on the development plan).

3. The third version digit will be updated when we fix serious bugs or accumulate some minor changes, including
   recipe related changes periodically (every two months or so).

## 4 Unit testing

ESPnet's testing is located under `test/`.  You can install additional packages for testing as follows:
``` console
$ cd <espnet_root>
$ . ./tools/activate_python.sh
$ pip install -e ".[test]"
```

### 4.1 Python

Then you can run the entire test suite using [flake8](http://flake8.pycqa.org/en/latest/), [autopep8](https://github.com/hhatto/autopep8), [black](https://github.com/psf/black) and [pytest](https://docs.pytest.org/en/latest/) with [coverage](https://pytest-cov.readthedocs.io/en/latest/reporting.html) by
``` console
./ci/test_python.sh
```
Followings are some useful tips when you are using pytest:
- To create a new test file. write functions named like `def test_yyy(...)` in files like `test_xxx.py` under `test/`.
[Pytest](https://docs.pytest.org/en/latest/) will automatically test them.
- Note that, [pytest-timeouts](https://pypi.org/project/pytest-timeouts/) raises **an error when any tests exceed 2.0 sec**. To keep unit tests fast, please avoid large parameters, dynamic imports, and file access. If your unit test really needs more time, you can annotate your test function with `@pytest.mark.timeout(sec)`.
- You can find pytest fixtures in `test/conftest.py`. [They finalize unit tests.](https://docs.pytest.org/en/latest/fixture.html#using-fixtures-from-classes-modules-or-projects)
- As it is important to make sure that the unit test covers more codes, we recommend you to use `pytest --cov-report term-missing --cov=<target_dir> tests/` to check the status of test coverage. For more details, please refer to [coverage-test](https://pytest-cov.readthedocs.io/en/latest/readme.html).
- For unit tests, we recommend you to separate large tests (e.g., `test_e2e_xxx.py`) into smaller ones, each one should test only methods/operations inside one file when it's possible. 
   

### 4.2 Bash scripts

You can also test the scripts in `utils` with [bats-core](https://github.com/bats-core/bats-core) and [shellcheck](https://github.com/koalaman/shellcheck).

To test:

``` console
./ci/test_bash.sh
```

## 5. Integration testing

Write new integration tests in [ci/test_integration.sh](ci/test_integration.sh) when you add new features in [espnet/bin](espnet/bin). They use our smallest dataset [egs/mini_an4](egs/mini_an4) to test `run.sh`. To make the coverage take them into account, don't forget `--python ${python}` support in your `run.sh`

```bash
# ci/integration_test.sh

python="coverage run --append"

cd egs/mini_an4/your_task
./run.sh --python "${python}"

```

### 5.1 Configuration files

- [setup.cfg](setup.cfg) configures pytest, black and flake8.
- [.travis.yml](.travis.yml) configures Travis-CI (unittests, doc deploy).
- [.circleci/config.yml](.circleci/config.yml) configures Circle-CI (unittests, integration tests).
- [.github/workflows](.github/workflows/) configures Github Actions (unittests, integration tests).

## 6 Writing new tools

You can place your new tools under
- `espnet/bin`: heavy and large (e.g., neural network related) core tools.
- `utils`: lightweight self-contained python/bash scripts.

For `utils` scripts, do not forget to add help messages and test scripts under `test_utils`.

### 8.1 Python tools guideline

To generate doc, do not forget `def get_parser(): -> ArgumentParser` in the main file.

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

### 8.2 Bash tools guideline

To generate doc, support `--help` to show its usage. If you use Kaldi's `utils/parse_option.sh`, define `help_message="Usage: $0 ..."`.


## Writing documentation

See [doc](doc/README.md).

## Adding pretrained models

Pack your trained models using `utils/pack_model.sh` and upload it [here](https://drive.google.com/open?id=1k9RRyc06Zl0mM2A7mi-hxNiNMFb_YzTF) (You require permission).
Add the shared link to `utils/recog_wav.sh` or `utils/synth_wav.sh` as follows:
```sh
    "tedlium.demo") share_url="https://drive.google.com/open?id=1UqIY6WJMZ4sxNxSugUqp3mrGb3j6h7xe" ;;
```
The model name is arbitrary for now.


## On CI failure

### Travis CI and Github Actions

1. read the log from PR checks > details

### Circle CI

1. read the log from PR checks > details
2. turn on Rerun workflow > Rerun job with SSH
3. open your local terminal and `ssh -p xxx xxx` (check circle ci log for the exact address)
4. try anything you can to pass the CI

### Codecov

1. write more tests to increase coverage
2. explain to reviewers why you can't increase coverage
