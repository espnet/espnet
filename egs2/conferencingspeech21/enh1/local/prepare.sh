#!/bin/bash

set -e
set -u
set -o pipefail


corpora_dir=
selected_list_dir=
outdir=

echo "$0 $*"
. utils/parse_options.sh


. ./path.sh || exit 1;
. ./db.sh || exit 1;


mkdir -p "${outdir}"
tmpdir=$(mktemp -d /tmp/cs21.XXXX)
trap 'rm -rf "$tmpdir"' EXIT

# prepare speech for training
for name in aishell_3 aishell_1 vctk; do
    python local/prepare_data_list.py \
        --outfile "${tmpdir}/train_${name}.lst" \
        --audiodirs "${corpora_dir}/${name}" \
        --audio-format "wav" \
        "${selected_list_dir}/train/${name}.name"
done

python local/prepare_data_list.py \
    --outfile "${tmpdir}/train_librispeech_360.lst" \
    --audiodirs "${corpora_dir}/librispeech_360" \
    --audio-format "flac" \
    "${selected_list_dir}/train/librispeech_360.name"

cat "${tmpdir}"/train_{aishell_3,aishell_1,vctk,librispeech_360}.lst > "${outdir}/train_clean.lst"

# prepare noise for training
python local/prepare_data_list.py \
    --outfile "${tmpdir}/musan.lst" \
    --audiodirs "${corpora_dir}/musan" \
    --audio-format "wav" \
    "${selected_list_dir}/train/musan.name"

python local/prepare_data_list.py \
    --outfile "${tmpdir}/audioset.lst" \
    --audiodirs "${corpora_dir}/audioset" \
    --audio-format "wav" \
    --ignore-missing-files True \
    "${selected_list_dir}/train/audioset.name"

cat "${tmpdir}"/{musan,audioset}.lst > "${outdir}/train_noise.lst"

# prepare speech for development
python local/prepare_data_list.py \
    --outfile "${outdir}/dev_clean.lst" \
    --audiodirs "${corpora_dir}/aishell_1" "${corpora_dir}/vctk" "${corpora_dir}/aishell_3" \
    --audio-format "wav" \
    "${selected_list_dir}/dev/clean.name"

# prepare noise for development
python local/prepare_data_list.py \
    --outfile "${outdir}/dev_noise.lst" \
    --audiodirs "${corpora_dir}/musan" \
    --audio-format "wav" \
    "${selected_list_dir}/dev/noise.name"

# Prepare the simulated RIR lists for training and development
for name in linear circle non_uniform; do
    for mode in train dev; do
        python local/prepare_data_list.py \
            --outfile "${outdir}/${mode}_${name}_rir.lst" \
            --audiodirs "${corpora_dir}/${name}" \
            --audio-format "wav" \
            "${selected_list_dir}/${mode}/${name}.name"
    done
done
