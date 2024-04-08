#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 (Xuankai Chang)Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo "$0 $*" >&2 # Print the command line for logging
. ./path.sh

nj=1
cmd=run.pl
nlsyms=""
lang=""
feat="" # feat.scp
oov="<unk>"
bpecode=""
allow_one_column=false
verbose=0
trans_type=char
filetype=""
preprocess_conf=""
category=""
out="" # If omitted, write in stdout
num_spkrs=1
help_message=$(cat << EOF
Usage: $0 <data-dir> <dict>
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
  --num-spkrs <num>                                # Number of speakers Default: 2
EOF
)
. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

dir=$1
dic=$2
tmpdir=$(mktemp -d ${dir}/tmp-XXXXX)
trap 'rm -rf ${tmpdir}' EXIT

# 1. Create scp files for inputs
#   These are not necessary for decoding mode, and make it as an option
input_strs=""
mkdir -p ${tmpdir}/input
if [ -n "${feat}" ]; then
    cat ${feat} > ${tmpdir}/input/feat.scp

    # Dump in the "legacy" style JSON format
    if [ -n "${filetype}" ]; then
        awk -v filetype=${filetype} '{print $1 " " filetype}' ${feat} \
            > ${tmpdir}/input/filetype.scp
    fi

    feat_to_shape.sh --cmd "${cmd}" --nj ${nj} \
        --filetype "${filetype}" \
        --preprocess-conf "${preprocess_conf}" \
        --verbose ${verbose} ${feat} ${tmpdir}/input/shape.scp

    input_strs=${input_strs}"--input-scps feat:${tmpdir}/input/feat.scp ${filetype:+filetype:${tmpdir}/input/filetype.scp} \
                             shape:${tmpdir}/input/shape.scp:shape "
fi

# 2. Create scp files for outputs
mkdir -p ${tmpdir}/output
output_strs=""
for outidx in $(seq ${num_spkrs}); do
  if [ ${num_spkrs} -eq 1 ]; then
    suffix=""
  else
    suffix="_spk"${outidx}
  fi
  if [ -n "${bpecode}" ]; then
      paste -d " " <(awk '{print $1}' ${dir}/text${suffix}) <(cut -f 2- -d" " ${dir}/text${suffix} \
          | spm_encode --model=${bpecode} --output_format=piece) \
          > ${tmpdir}/output/token${suffix}.scp
  elif [ -n "${nlsyms}" ]; then
      text2token.py -s 1 -n 1 -l ${nlsyms} ${dir}/text${suffix} > ${tmpdir}/output/token${suffix}.scp
  else
      text2token.py -s 1 -n 1 ${dir}/text${suffix} > ${tmpdir}/output/token${suffix}.scp
  fi
  < ${tmpdir}/output/token${suffix}.scp utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} > ${tmpdir}/output/tokenid${suffix}.scp
  # +2 comes from CTC blank and EOS
  vocsize=`tail -n 1 ${dic} | awk '{print $2}'`
  odim=`echo "$vocsize + 2" | bc`
  < ${tmpdir}/output/tokenid${suffix}.scp awk -v odim=${odim} '{print $1 " " NF-1 "," odim}' > ${tmpdir}/output/shape${suffix}.scp

  cat ${dir}/text${suffix} > ${tmpdir}/output/text${suffix}.scp

  # 3. Create scp files for the others
  mkdir -p ${tmpdir}/other
  if [ -n "${lang}" ]; then
      awk -v lang=${lang} '{print $1 " " lang}' ${dir}/text${suffix} > ${tmpdir}/other/lang${suffix}.scp
  fi

  output_strs=${output_strs}"--output-scps text:${tmpdir}/output/text${suffix}.scp \
                             token:${tmpdir}/output/token${suffix}.scp \
                             tokenid:${tmpdir}/output/tokenid${suffix}.scp \
                             shape:${tmpdir}/output/shape${suffix}.scp:shape "
done

if [ -n "${category}" ]; then
    awk -v category=${category} '{print $1 " " category}' ${dir}/text${suffix} \
        > ${tmpdir}/other/category.scp
fi
cat ${dir}/utt2spk > ${tmpdir}/other/utt2spk.scp

# 4. Merge scp files into a JSON file
opts="${input_strs} ${output_strs} "
for intype in other; do
    if [ -z "$(find "${tmpdir}/${intype}" -name "*.scp")" ]; then
        continue
    fi

    if [ ${intype} != other ]; then
        opts+="--${intype}-scps "
    else
        opts+="--scps "
    fi

    for x in "${tmpdir}/${intype}"/*.scp; do
        k=$(basename ${x} .scp)
        if [ ${k} = shape ]; then
            opts+="shape:${x}:shape "
        else
            opts+="${k}:${x} "
        fi
    done
done

if ${allow_one_column}; then
    opts+="--allow-one-column true "
else
    opts+="--allow-one-column false "
fi

if [ -n "${out}" ]; then
    opts+="-O ${out}"
fi
merge_scp2json.py --verbose ${verbose} ${opts}
