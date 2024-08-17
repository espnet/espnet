# ESPnet document generation

## Install

We use [sphinx](https://www.sphinx-doc.org) to generate HTML documentation.

```sh
# Clean conda env for docs
$ cd <espnet_root>
$ conda create -p ./envs python=3.8
$ conda activate ./envs

# Requirements
$ pip install -e ".[doc]"
$ conda install conda-forge::ffmpeg
$ conda install conda-forge::nodejs

# (Optional requirement) To use flake8-docstrings
$ pip install -U flake8-docstrings
```

If you used the above clean conda environment, you have write your own `. tools/activate_python.sh`.
The example will be:
```sh
#!/usr/bin/env bash

. <conda_root>/miniconda/etc/profile.d/conda.sh && conda activate <espnet_root>/envs
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

You can generate and test the webpage using sphinx Makefile.
```sh
$ cd <espnet_root>
$ ./ci/doc.sh
$ npm run docs:dev
```

## Deploy

When your PR is merged into `master` branch, our [CI](https://github.com/espnet/espnet/blob/master/.github/workflows/doc.yml) will automatically deploy your sphinx html into https://espnet.github.io/espnet/.
