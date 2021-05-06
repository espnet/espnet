#!/usr/bin/env bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

cases=""
speeds="0.9 1.0 1.1"
langs=""
write_utt2num_frames=true
nj=32
cmd=""

help_message=$(cat <<EOF
Usage: $0 [options] <data-dir> <destination-dir> <fbankdir>
e.g.: $0 data/train en de
Options:
  --cases                              # target case information (e.g., lc.rm, lc, tc)
  --speeds                             # speed used in speed perturbation (e.g., 0.9. 1.0, 1.1)
  --langs                              # all languages (source + target)
  --write_utt2num_frames               # write utt2num_frames in steps/make_fbank_pitch.sh
  --cmd <run.pl|queue.pl <queue opts>> # how to run jobs
  --nj <nj>                            # number of parallel jobs
EOF
)
echo "$0 $*"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

data_dir=$1
dst=$2
fbankdir=$3

tmpdir=$(mktemp -d ${data_dir}/tmp-XXXXX)
trap 'rm -rf ${tmpdir}' EXIT

for sp in ${speeds}; do
    utils/perturb_data_dir_speed.sh ${sp} ${data_dir} ${tmpdir}/temp.${sp}
done
utils/combine_data.sh --extra-files utt2uniq ${dst} ${tmpdir}/temp.*

steps/make_fbank_pitch.sh --cmd ${cmd} --nj ${nj} --write_utt2num_frames ${write_utt2num_frames} \
    ${dst} exp/make_fbank/"$(basename ${dst})" ${fbankdir}
utils/fix_data_dir.sh ${dst}
utils/validate_data_dir.sh --no-text ${dst}

if [ -n "${langs}" ]; then
    # for ST/MT recipe + ASR recipe in ST recipe
   for lang in ${langs}; do
        for case in ${cases}; do
            if [ -f ${dst}/text.${case}.${lang} ]; then
                rm ${dst}/text.${case}.${lang}
            fi
        done
        touch ${dst}/text.${case}.${lang}

        for sp in ${speeds}; do
            awk -v p="sp${sp}-" '{printf("%s %s%s\n", $1, p, $1);}' ${data_dir}/utt2spk > ${dst}/utt_map

            for case in ${cases}; do
                utils/apply_map.pl -f 1 ${dst}/utt_map <${data_dir}/text.${case}.${lang} >> ${dst}/text.${case}.${lang}
            done
        done
    done
else
    # for ASR only recipe
    touch ${dst}/text
    for sp in ${speeds}; do
        awk -v p="sp${sp}-" '{printf("%s %s%s\n", $1, p, $1);}' ${data_dir}/utt2spk > ${dst}/utt_map
        utils/apply_map.pl -f 1 ${dst}/utt_map <${data_dir}/text >>${dst}/text
    done
fi

rm -rf ${tmpdir}*
