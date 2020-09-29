# How to contribute to ESPnet

If you are interested in contributing to ESPnet, your contributions will fall into three categories:

1. If you want to propose a new feature and implement it, please post about your intended feature at the issues, 
   or you can contact Shinji Watanabe <shinjiw@ieee.org> or other main developers. 
   We shall discuss the design and implementation.
   Once we agree that the plan looks good, go ahead and implement it.
   You can find ongoing major development plans at https://github.com/espnet/espnet/milestones

2. If you want to implement a minor feature or bug-fix for an issue, please first take a look at 
   the existing issues (https://github.com/espnet/espnet/pulls) and/or pull requests (https://github.com/espnet/espnet/pulls).
   Pick an issue and comment on the task that you want to work on this feature.
   If you need more context on a particular issue, please ask us and then we shall provide more information.
   
   We also welcome if you find some bugs during your actual use of ESPnet and make a PR to fix them.

3. ESPnet provides and maintains a lot of reproducible examples similar to Kaldi (called `recipe`).
   The recipe creation/update/bug-fix is one of our major development items, and we really encourage 
   you to work on it.
   When you port a Kaldi recipe to ESPnet, see https://github.com/espnet/espnet/wiki/How-to-port-the-Kaldi-recipe-to-the-ESPnet-recipe%3F 
   
   We also encourage you to report your results with your detailed environmental info and upload the model for the reproducibility 
   (e.g., see https://github.com/espnet/espnet/blob/master/egs/tedlium2/asr1/RESULTS.md).
   
   To make a report for `RESULTS.md`
	 - execute `show_result.sh` at a recipe main directory (where `run.sh` is located), as follows. 
	   You'll get environmental information and the evaluation result of each experiments in a markdown format.
	   ```
	   $ show_result.sh
	   ```
	 - execute `pack_model.sh` at a recipe main directory as follows. You'll get model information in a markdown format
	   ```
	   $ pack_model.sh --lm <language model> <tr_conf> <dec_conf> <cmvn> <e2e>
	   ```
	 - `pack_model.sh` also produces a packed espnet model (`model.tar.gz`). If you upload this model to somewhere with a download link,
	   please put the link information to `RESULTS.md`.
	 - please contact Shinji Watanabe <shinjiw@ieee.org> if you want a web storage to put your model files.

Once you finish implementing a feature or bugfix, please send a Pull Request to https://github.com/espnet/espnet

If you are not familiar with creating a Pull Request, here are some guides:

- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/

## Version policy and development branches

We basically develop in the `master` branch.

1. We will keep the first version digit `0` until we have some super major changes in the project organization level.

2. The second version digit will be updated when we have major updates including new functions and refactoring, and 
   their related bug fix and recipe changes.
   This version update will be done roughly at every half year so far (but it depends on the development plan).

3. The third version digit will be updated when we fix serious bugs or accumulate some minor changes including
   recipe related changes periodically (every two months or so).

## Unit testing

ESPnet's testing is located under `test/`.  You can install additional packages for testing as follows:
``` console
$ cd <espnet_root>
$ pip install -e ".[test]"
```

### Python

Then you can run the entire test suite using [flake8](http://flake8.pycqa.org/en/latest/), [autopep8](https://github.com/hhatto/autopep8), [black](https://github.com/psf/black) and [pytest](https://docs.pytest.org/en/latest/) with [coverage](https://pytest-cov.readthedocs.io/en/latest/reporting.html) by
``` console
./ci/test_python.sh
```

To create new test file. write functions named like `def test_yyy(...)` in files like `test_xxx.py` under `test/`.
[Pytest](https://docs.pytest.org/en/latest/) will automatically test them. Note that, [pytest-timeouts](https://pypi.org/project/pytest-timeouts/) raises **an error when any tests exceed 2.0 sec**. To keep unittests fast, please avoid large parameters, dynamic imports, and file access. If your unittest really needs more time, you can annotate your test function with `@pytest.mark.timeout(sec)`. You can find pytest fixtures in `test/conftest.py`. [They finalize unit tests.](https://docs.pytest.org/en/latest/fixture.html#using-fixtures-from-classes-modules-or-projects)

### Bash scripts

You can also test the scripts in `utils` with [bats-core](https://github.com/bats-core/bats-core) and [shellcheck](https://github.com/koalaman/shellcheck).

To test:

``` console
./ci/test_bash.sh
```

## Integration testing

Write new integration tests in [ci/test_integration.sh](ci/test_integration.sh) when you add new features in [espnet/bin](espnet/bin). They use our smallest dataset [egs/mini_an4](egs/mini_an4) to test `run.sh`. To make the coverage take them into account, don't forget `--python ${python}` support in your `run.sh`

```bash
# ci/integration_test.sh

python="coverage run --append"

cd egs/mini_an4/your_task
./run.sh --python "${python}"

```

### Configuration files

- [setup.cfg](setup.cfg) configures pytest, black and flake8.
- [.travis.yml](.travis.yml) configures Travis-CI (unittests, doc deploy).
- [.circleci/config.yml](.circleci/config.yml) configures Circle-CI (unittests, integration tests).
- [.github/workflows](.github/workflows/) configures Github Actions (unittests, integration tests).

## Writing new tools

You can place your new tools under
- `espnet/bin`: heavy and large (e.g., neural network related) core tools.
- `utils`: lightweight self-contained python/bash scripts.

For `utils` scripts, do not forget to add help messages, and test scripts under `test_utils`.

### Python tools guideline

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

### Bash tools guideline

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

1. read log from PR checks > details

### Circle CI

1. read log from PR checks > details
2. turn on Rerun workflow > Rerun job with SSH
3. open your local terminal and `ssh -p xxx xxx` (check circle ci log for the exact address)
4. try anything you can to pass the CI

### Codecov

1. write more tests to increase coverage
2. explain to reviewers why you can't increase coverage
