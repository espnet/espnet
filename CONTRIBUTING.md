# How to contribute to ESPnet

If you are interested in contributing to ESPnet, your contributions will fall into two categories:

1. You want to propose a new Feature and implement it
   post about your intended feature, and we shall discuss the design and implementation.
   Once we agree that the plan looks good, go ahead and implement it.
   You can find ongoing major development plans at https://github.com/espnet/espnet/milestones

2. You want to implement a feature or bug-fix for an outstanding issue
   Look at the outstanding issues here: https://github.com/espnet/espnet/issues
   Pick an issue and comment on the task that you want to work on this feature
   If you need more context on a particular issue, please ask us and then we shall provide more information.

Once you finish implementing a feature or bugfix, please send a Pull Request to https://github.com/espnet/espnet

If you are not familiar with creating a Pull Request, here are some guides:

- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/

# Version policy and development branches

We basically maintain the `master` and `v.0.X.0` branches for our major developments.

1. We will keep the first version digit `0` until we have some super major changes in the project organization level.

2. The second version digit will be updated when we have major updates including new functions and refactoring, and 
   their related bug fix and recipe changes.
   This version update will be done roughly at every half year so far (but it depends on the development plan).
   This is developed at the `v.0.X.0` branch to avoid confusions in the `master` branch.

3. The third version digit will be updated when we fix serious bugs or accumulate some minor changes including
   recipe related changes periodically (every two months or so).
   This is developed at the `master` branch, and these changes are also reflected to the `v.0.X.0` branch frequently.

## Developing locally with ESPnet

TBD

## Unit testing

ESPnet's testing is located under `test/`.  You can install additional packages for testing as follows:
``` console
$ cd <espnet_root>
$ pip install -e ".[test]"
```

Then you can run the entire test suite with
``` console
$ pytest
```

To create new test file. write functions named like `def test_yyy(...)` in files like `test_xxx.py` under `test/`.
[Pytest](https://docs.pytest.org/en/latest/) will automatically test them.

We also recommend you to follow our coding style that can be checked as
``` console
$ flake8 espnet test
$ autopep8 -r espnet test --global-config .pep8 --diff --max-line-length 120 | tee check_autopep8
$ test ! -s check_autopep8
```

You can find pytest fixtures in `test/conftest.py`. [They finalize unit tests.](https://docs.pytest.org/en/latest/fixture.html#using-fixtures-from-classes-modules-or-projects)

You can also test the scripts in `utils` with [bats-core](https://github.com/bats-core/bats-core) and [shellcheck](https://github.com/koalaman/shellcheck).

To test:

``` console
./ci/test_bash.sh
```

### Configuration files

- [setup.cfg](setup.cfg) configures pytest and flake8.
- [.travis.yml](.travis.yml) configures Travis-CI.


## Writing documentation

See [doc](doc/README.md).

## Adding pretrained models

Pack your trained models using `utils/pack_model.sh` and upload it [here](https://drive.google.com/open?id=1k9RRyc06Zl0mM2A7mi-hxNiNMFb_YzTF) (You require permission).
Add the shared link to `utils/recog_wav.sh` as follows:
```sh
    "tedlium.demo") share_url="https://drive.google.com/open?id=1UqIY6WJMZ4sxNxSugUqp3mrGb3j6h7xe" ;;
```
The model name is arbitrary for now.


## Python 2 and 3 portability tips

See matplotlib's guideline https://matplotlib.org/devel/portable_code.html
We do not block your PR even if it is not portable.


## On CI failure

### Travis CI

1. read log from PR checks > details

### Circle CI

1. read log from PR checks > details
2. turn on Rerun workflow > Rerun job with SSH
3. open your local terminal and `ssh -p xxx xxx` (check circle ci log for the exact address)
4. try anything you can to pass the CI

