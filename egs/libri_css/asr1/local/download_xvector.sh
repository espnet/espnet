#!/usr/bin/env bash
#
# Copyright  2020  University of Stuttgart (Author: Pavel Denisov)
# Apache 2.0

# Begin configuration section.
# End configuration section
. ./utils/parse_options.sh  # accept options

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 1 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 <diarizer-dir>"
  echo -e >&2 "eg:\n  $0 download/xvector_voxceleb"
  exit 1
fi

diarizer_dir=$1

set -e -o pipefail

mkdir -p ${diarizer_dir}
cd ${diarizer_dir}

# Download x-vector extractor trained on VoxCeleb2 data
wget http://kaldi-asr.org/models/12/0012_diarization_v1.tar.gz
tar -xvzf 0012_diarization_v1.tar.gz
rm -f 0012_diarization_v1.tar.gz

# Download PLDA model trained on augmented Librispeech data
rm 0012_diarization_v1/exp/xvector_nnet_1a/plda
wget https://desh2608.github.io/static/files/jsalt/plda -P 0012_diarization_v1/exp/xvector_nnet_1a/
