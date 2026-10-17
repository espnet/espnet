#!/usr/bin/env bash
set -euo pipefail

python=python3
train_out=dump/raw/train_all_no_filter_lang
. ./path.sh
. utils/parse_options.sh

if [ ! -d dump/raw/train_babel_over_10s_lang ]; then
    bash local/filter_babel_over_10s.sh --python "${python}"
fi

${python} local/combine_lid_data.py --output-dir "${train_out}" \
    dump/raw/train_fleurs_lang \
    dump/raw/train_ml_superb2_lang \
    dump/raw/train_voxlingua107_lang \
    dump/raw/train_voxpopuli_lang \
    dump/raw/train_babel_over_10s_lang
