#!/usr/bin/env bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

maxframes=3000
maxchars=400
utt_extra_files="text.tc text.lc text.lc.rm"
no_feat=false

help_message=$(cat <<EOF
Usage: $0 [options] <data-dir> <langs>
e.g.: $0 data/train "en de"
Options:
  --maxframes        # number of maximum input frame length
  --maxchars         # number of maximum character length
  --utt_extra_files  # extra text files for target sequence
  --no_feat          # set to True for MT recipe
EOF
)
echo "$0 $*"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

data_dir=$1
langs=$2

mkdir -p ${data_dir}
tmpdir=$(mktemp -d ${data_dir}/tmp-XXXXX)
trap 'rm -rf ${tmpdir}' EXIT

# remove utt having more than ${maxframes} frames
# remove utt having more than ${maxchars} characters
for lang in ${langs}; do
    remove_longshortdata.sh --no_feat ${no_feat} --maxframes ${maxframes} --maxchars ${maxchars} ${data_dir}.${lang} ${tmpdir}.${lang}
done

# Match the number of utterances between source and target languages
for lang in ${langs}; do
    cut -f 1 -d " " ${tmpdir}.${lang}/text > ${tmpdir}.${lang}/reclist
    if [ ! -f ${tmpdir}/reclist ]; then
        cp ${tmpdir}.${lang}/reclist  ${tmpdir}/reclist
    else
        # extract common lines
        comm -12 ${tmpdir}/reclist ${tmpdir}.${lang}/reclist > ${tmpdir}/reclist.tmp
        mv ${tmpdir}/reclist.tmp ${tmpdir}/reclist
    fi
done

for lang in ${langs}; do
    reduce_data_dir.sh ${tmpdir}.${lang} ${tmpdir}/reclist ${data_dir}.${lang}
    utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${data_dir}.${lang}
done

rm -rf ${tmpdir}*
