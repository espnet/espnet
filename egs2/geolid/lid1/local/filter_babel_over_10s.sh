#!/usr/bin/env bash
set -euo pipefail

dump_dir="dump/raw"
splits="train_babel_lang dev_babel_lang"
python=python3
. utils/parse_options.sh || exit 1

for split in ${splits}; do
    target="${split%_lang}_over_10s_lang"
    ${python} local/filter_babel_over_10s.py \
        --babel_dir "${dump_dir}/${split}" \
        --babel_over_10s_dir "${dump_dir}/${target}"
done
