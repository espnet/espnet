#!/usr/bin/env bash

set -euo pipefail

flake8 espnet test utils;
autopep8 -r espnet test utils --global-config .pep8 --diff --max-line-length 120 | tee check_autopep8
test ! -s check_autopep8
pytest
