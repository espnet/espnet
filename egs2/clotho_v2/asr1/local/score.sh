#!/usr/bin/env bash
# Copyright 2021 Carnegie Mellon University (Author : Roshan Sharma)

# TODO(sbharad2): Change for your own use case.
## begin configuration section.
# end configuration section.


[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;


if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-inference-dir>"
  exit 1;
fi

# pip install aac-metrics
# aac-metrics-download
asr_expdir=$1
splits=(evaluation validation)

set -ex
for split in ${splits[@]}; do
	for decode_file in $((ls -d ${asr_expdir}/*/*/* && ls -d ${asr_expdir}/*/*) | grep "${split}/text"); do
		echo "Decode file: ${decode_file} for split:${split}"
		python local/evaluation.py --decode_file=${decode_file} --split=${split}
	done
done