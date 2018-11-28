#!/bin/bash

# Copyright 2018, Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

# Config:
print_clean=false
print_nf=false
print_2ch=false

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

if ${print_clean}; then
    echo "RESULTS - Cln"
    local/score_for_reverb_cln.sh --wer true --nlsyms ${nlsyms} \
        		      "${expdir}/decode_*_cln_${decode_part_dir}/data.json" \
        		      ${dict} ${expdir}/decode_summary_cln_${decode_part_dir}
    echo ""
fi
if ${print_nf}; then
    echo "RESULTS - 1ch - No Front End"
    local/score_for_reverb.sh --wer true --nlsyms ${nlsyms} \
        		      "${expdir}/decode_*_1ch_${decode_part_dir}/data.json" \
        		      ${dict} ${expdir}/decode_summary_1ch_${decode_part_dir}
    echo ""
fi
echo "RESULTS - 1ch - WPE"
local/score_for_reverb.sh --wer true --nlsyms ${nlsyms} \
    		      "${expdir}/decode_*_1ch_wpe_${decode_part_dir}/data.json" \
    		      ${dict} ${expdir}/decode_summary_1ch_wpe_${decode_part_dir}
echo ""
if ${print_2ch}; then
    echo "RESULTS - 2ch - WPE+BeamformIt"
    local/score_for_reverb.sh --wer true --nlsyms ${nlsyms} \
        		      "${expdir}/decode_*_2ch_beamformit_${decode_part_dir}/data.json" \
        		      ${dict} ${expdir}/decode_summary_2ch_beamformit_${decode_part_dir}
    echo ""
fi
echo "RESULTS - 8ch - WPE+BeamformIt"
local/score_for_reverb.sh --wer true --nlsyms ${nlsyms} \
		      "${expdir}/decode_*_8ch_beamformit_${decode_part_dir}/data.json" \
		      ${dict} ${expdir}/decode_summary_8ch_beamformit_${decode_part_dir}
