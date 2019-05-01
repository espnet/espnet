#!/bin/bash

# Copyright 2019 Nagoya University (Koichi Miyazaki)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
echo "$0 $*" >&2 # Print the command line for logging
. ./path.sh

nj=1
cmd=run.pl
train_feat="" # feat.scp for training
validation_feat="" # feat.scp for validation
label="" # metadata directory e.g. ./DCASE2019_task4/dataset/metadata
verbose=0
. utils/parse_options.sh
if [ $# != 1 ]; then
    cat << EOF 1>&2
Usage: $0 <data-dir>
e.g. $0 data/train data/lang_1char/train_units.txt
Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --feat <feat-scp>                                # feat.scp
  --oov <oov-word>                                 # Default: <unk>
  --out <outputfile>                               # If omitted, write in stdout
  --filetype <mat|hdf5|sound.hdf5>                 # Specify the format of feats file
  --preprocess-conf <json>                         # Apply preprocess to feats when creating shape.scp
  --verbose <num>                                  # Default: 0
EOF
    exit 1;
fi

# FIXME: conda warning; unbound variable
#set -euo pipefail

dir=$1
tmpdir=$(mktemp -d ${dir}/tmp-XXXXX)
trap 'rm -rf ${tmpdir}' EXIT

# 1. Create scp files for inputs
#   These are not necessary for decoding mode, and make it as an option
mkdir -p ${tmpdir}/input
if [ -n "${train_feat}" ]; then
    for label_type in synthetic unlabel_in_domain weak; do
        grep ^${label_type} ${train_feat} > ${tmpdir}/input/feats_${label_type}.scp
    done
fi

if [ -n "${validation_feat}" ]; then
    for label_type in eval_dcase2018 test_dcase2018 validation; do
        grep ^${label_type} ${validation_feat} > ${tmpdir}/input/feats_${label_type}.scp
    done
fi

# 2. Create scp files for outputs
mkdir -p ${tmpdir}/output

if [ -n "${label}" ]; then
    for x in train validation; do
        if [ ${x} = train ]; then
            for label_type in synthetic weak; do
                csv=${label}/${x}/${label_type}.csv
                if [ ${label_type} = synthetic ]; then
                    for id in $(tail -n +2 ${csv} | awk '{print $1}' | uniq); do
                        echo -n ${label_type}-$(basename $id .wav)
                        cat $csv | grep ^${id} | awk '{printf " %s %s %s", $2, $3, $4}'
                        echo ""
                    done > ${tmpdir}/output/label_${label_type}.scp
                else
                    for id in $(tail -n +2 ${csv} | awk '{print $1}' | uniq); do
                        echo -n ${label_type}-$(basename $id .wav)
                        cat $csv | grep ^${id} | awk '{printf " %s", $2}'
                        echo ""
                    done > data/label_${label_type}.scp
                fi
            done
        elif [ ${x} = validation ]; then
            for label_type in eval_dcase2018 test_dcase2018 validation; do
                csv=${label}/${x}/${label_type}.csv
                for id in $(tail -n +2 ${csv} | awk '{print $1}' | uniq); do
                    echo -n ${label_type}-$(basename $id .wav)
                    cat $csv | grep ^${id} | awk '{printf " %s %s %s", $2, $3, $4}'
                    echo ""
                done > ${tmpdir}/output/label_${label_type}.scp
            done
        fi
    done
fi

# 3. Merge scp files into a JSON file
for x in train validation; do
    if [ ${x} = train ]; then
        for label_type in synthetic unlabel_in_domain; do
            opts=""
            opts+="--input-scps "
            opts+="feats:${tmpdir}/input/feats_${label_type}.scp "
            if [ ${label_type} != unlabel_in_domain ]; then
                sort ${tmpdir}/input/feats_${label_type}.scp -o ${tmpdir}/input/feats_${label_type}.scp
                sort ${tmpdir}/output/label_${label_type}.scp -o ${tmpdir}/output/label_${label_type}.scp
                opts+="--output-scps "
                opts+="label:${tmpdir}/output/label_${label_type}.scp "
            fi
            ./local/merge_scp2json.py --verbose ${verbose} ${opts} > ./data/${x}/data_${label_type}.json
        done
    elif [ ${x} = validation ]; then
        for label_type in eval_dcase2018 test_dcase2018 validation; do
            opts=""
            opts+="--input-scps "
            opts+="feats:${tmpdir}/input/feats_${label_type}.scp "
            if [ ${label_type} != unlabel_in_domain ]; then
                sort ${tmpdir}/input/feats_${label_type}.scp -o ${tmpdir}/input/feats_${label_type}.scp
                sort ${tmpdir}/output/label_${label_type}.scp -o ${tmpdir}/output/label_${label_type}.scp
                opts+="--output-scps "
                opts+="label:${tmpdir}/output/label_${label_type}.scp "
            fi
            ./local/merge_scp2json.py --verbose ${verbose} ${opts} > ./data/${x}/data_${label_type}.json
        done
    fi
done

rm -fr ${tmpdir}
