#!/usr/bin/env bash

. tools/activate_python.sh

set -euo pipefail

# TODO: remove files from this list!
flake8_black_list=

n_blacklist=$(wc -l <<< "${flake8_black_list}")
n_all=$(find espnet -name "*.py" | wc -l)
n_ok=$((n_all - n_blacklist))
cov=$(echo "scale = 4; 100 * ${n_ok} / ${n_all}" | bc)
echo "flake8-docstrings ready files coverage: ${n_ok} / ${n_all} = ${cov}%"

# --extend-ignore for wip files for flake8-docstrings
flake8 --show-source --extend-ignore=D test utils doc ${flake8_black_list} espnet2 test/espnet2 egs/*/*/local/*.py

# white list of files that should support flake8-docstrings
flake8 --show-source espnet --exclude=${flake8_black_list//$'\n'/,}
