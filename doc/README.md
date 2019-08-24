# ESPnet document generation

## Install

We use [travis-sphinx](https://github.com/Syntaf/travis-sphinx) to generate & deploy HTML documentation.

```sh
$ cd <espnet_root>
$ pip install -e ".[doc]"
$ pip install -U flake8-docstrings
```

## Style check using flake8-docstrings

You can check that your docstring style is correct by `ci/test_flake8.sh` using [flake8-docstrings](https://pypi.org/project/flake8-docstrings/).
Note that many existing files have been added to **the black list** in the script to avoid this check by default.
You can freely remove files that you wanna improve.

```bash
# in ci/test_flake8.sh

# you can improve poorly written docstrings from here
flake8_black_list="\
espnet/__init__.py
espnet/asr/asr_mix_utils.py
espnet/asr/asr_utils.py
espnet/asr/chainer_backend/asr.py
...
"

# --extend-ignore for wip files for flake8-docstrings
flake8 --extend-ignore=D test utils doc ${flake8_black_list}

# white list of files that should support flake8-docstrings
flake8 espnet --exclude=${flake8_black_list//$'\n'/,}
```

DO NOT ADD NEW FILES TO THIS BLACK LIST!

## Generate HTML

You can generate local HTML manually using sphinx Makefile

```sh
$ cd <espnet_root>
$ ./ci/doc.sh
```

open `doc/build/html/index.html`

## Deploy

When your PR is merged into `master` branch, our [Travis-CI](https://github.com/espnet/espnet/blob/master/.travis.yml) will automatically deploy your sphinx html into https://espnet.github.io/espnet/ by `travis-sphinx deploy`.
