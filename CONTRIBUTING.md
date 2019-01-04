# How to contribute to ESPnet

If you are interested in contributing to ESPnet, your contributions will fall into two categories:

1. You want to propose a new Feature and implement it
   post about your intended feature, and we shall discuss the design and implementation.
   Once we agree that the plan looks good, go ahead and implement it.
        
2. You want to implement a feature or bug-fix for an outstanding issue
   Look at the outstanding issues here: https://github.com/espnet/espnet/issues
   Pick an issue and comment on the task that you want to work on this feature
   If you need more context on a particular issue, please ask us and then we shall provide more information.

Once you finish implementing a feature or bugfix, please send a Pull Request to https://github.com/espnet/espnet

If you are not familiar with creating a Pull Request, here are some guides:

- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/


## Developing locally with ESPnet

TBD

## Unit testing

ESPnet's testing is located under `test/`.  You can install additional packages for testing as follows:
``` console
$ pip install -r tools/test_requirements.txt
```

Then you can run the entire test suite with
``` console
$ pytest
```

To create new test file. write functions named like `def test_yyy(...)` in files like `test_xxx.py` under `test/`.
[Pytest](https://docs.pytest.org/en/latest/) will automatically test them.

We also recommend you to follow our coding style that can be checked as
``` console
$ flake8 src test
$ autopep8 -r src test --exclude src/utils --global-config .pep8 --diff --max-line-length 120 | tee check_autopep8
$ test ! -s check_autopep8
```

You can find pytest fixtures in `test/conftest.py`. [They finalize unit tests.](https://docs.pytest.org/en/latest/fixture.html#using-fixtures-from-classes-modules-or-projects)

### Configuration files

- [setup.cfg](setup.cfg) configures pytest and flake8.
- [.travis.yml](.travis.yml) configures Travis-CI.


## Writing documentation

See [doc](doc/README.md).

## Python 2 and 3 portability tips

See matplotlib's guideline https://matplotlib.org/devel/portable_code.html
We do not block your PR even if it is not portable.
