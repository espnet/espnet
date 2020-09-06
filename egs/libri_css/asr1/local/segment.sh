#!/usr/bin/env bash
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Copyright  2020  University of Stuttgart (Author: Pavel Denisov)
# Apache 2.0

# Begin configuration section.
score_sad=true
# End configuration section
. ./utils/parse_options.sh  # accept options

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 1 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 <data-dir>"
  echo -e >&2 "eg:\n  $0 data/eval"
  exit 1
fi

data_dir=$1

set -e -o pipefail

# Ensure that webrtcvad is installed
if [ -z `pip freeze | grep webrtcvad` ]; then
  pip install webrtcvad
fi

# Perform SAD using py-webrtcvad package

if [ ! -f ${data_dir}/wav.scp ]; then
  echo "$0: Not performing SAD on ${data_dir}"
  exit 0
fi

# Perform segmentation
local/segmentation/apply_webrtcvad.py --mode 0 $data_dir | sort > $data_dir/segments

# Create dummy utt2spk file from obtained segments
awk '{print $1, $2}' ${data_dir}/segments > ${data_dir}/utt2spk
utils/utt2spk_to_spk2utt.pl ${data_dir}/utt2spk > ${data_dir}/spk2utt

# Generate RTTM file from segmentation performed by SAD. This can
# be used to evaluate the performance of the SAD as an intermediate
# step.
steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
${data_dir}/utt2spk ${data_dir}/segments ${data_dir}/rttm

if [ $score_sad == "true" ]; then
  echo "Scoring $datadir.."
  # We first generate the reference RTTM from the backed up utt2spk and segments
  # files.
  ref_rttm=${data_dir}/ref_rttm
  steps/segmentation/convert_utt2spk_and_segments_to_rttm.py ${data_dir}/utt2spk.bak \
  ${data_dir}/segments.bak ${data_dir}/ref_rttm

  md-eval.pl -r $ref_rttm -s ${data_dir}/rttm |\
    awk 'or(/MISSED SPEECH/,/FALARM SPEECH/)'
fi
