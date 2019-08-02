#!/bin/bash

# Copyright 2018, Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

# Config:
print_pipeline=false

. utils/parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Wrong #arguments ($#, expected 4)"
   echo "Usage: local/get_results.sh [options] <nlsyms> <dict> <expdir> <decode_part_dir>"
   exit 1;
fi

nlsyms=$1
dict=$2
expdir=$3
decode_part_dir=$4

if $print_pipeline; then
    echo "RESULTS - 8ch - Pipeline - WPE + BeamformIt"
    local/score_for_reverb.sh --wer true --nlsyms ${nlsyms} \
		      "${expdir}/decode_*_8ch_wpe_beamformit_${decode_part_dir}/data.json" \
		      ${dict} ${expdir}/decode_summary_8ch_pipeline_${decode_part_dir}
fi

echo "RESULTS - 8ch - WPE + MVDR"
local/score_for_reverb.sh --wer true --nlsyms ${nlsyms} \
		      "${expdir}/decode_*_8ch_multich_${decode_part_dir}/data.json" \
		      ${dict} ${expdir}/decode_summary_8ch_multich_${decode_part_dir}
