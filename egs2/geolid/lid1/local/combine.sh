#!/usr/bin/env bash
set -euo pipefail

python=python3
dump_dir=dump/raw
train_out=
. ./path.sh
. utils/parse_options.sh
train_out=${train_out:-${dump_dir}/train_all_no_filter_lang}

if [ ! -d "${dump_dir}/train_babel_over_10s_lang" ]; then
    bash local/filter_babel_over_10s.sh --dump_dir "${dump_dir}" --python "${python}"
fi

${python} local/combine_lid_data.py --output-dir "${train_out}" \
    "${dump_dir}/train_fleurs_lang" \
    "${dump_dir}/train_ml_superb2_lang" \
    "${dump_dir}/train_voxlingua107_lang" \
    "${dump_dir}/train_voxpopuli_lang" \
    "${dump_dir}/train_babel_over_10s_lang"

# Keep all classification labels; URIEL has no geographic target for Toki Pona.
${python} local/prepare_lang2vec.py \
    --lang2utt "${train_out}/lang2utt" \
    --output data/combined_lang2vec.txt --exclude tok
