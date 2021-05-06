#!/usr/bin/env bash
#
# Copyright  2020  University of Stuttgart (Author: Pavel Denisov)
# Apache 2.0

# Begin configuration section.
# End configuration section
. ./utils/parse_options.sh  # accept options

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 2 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 <in-data-dir> <out-data-dir>"
  echo -e >&2 "eg:\n  $0 data/eval data/eval_oracle"
  exit 1
fi

data_in=$1
data_out=$2

set -e -o pipefail

mkdir -p ${data_out}

cp ${data_in}/wav.scp ${data_out}/

# .bak files were created by local/data_prep_mono.sh
cp ${data_in}/segments.bak ${data_out}/segments
cp ${data_in}/utt2spk.bak ${data_out}/utt2spk
cp ${data_in}/text.bak ${data_out}/text
utils/utt2spk_to_spk2utt.pl ${data_out}/utt2spk > ${data_out}/spk2utt
