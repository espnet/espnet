#!/bin/bash
# This script is used to run pyannote diarization pipeline.
set -euo pipefail

in_dir=
regex=
out_dir=


. ./path.sh
. parse_options.sh

# performing diarization on multiple mics
python ./local/pyannote_diarize.py --in_dir $input_dir --regex $regex --out_dir $out_dir
# using dover-lap to combine all hypotheses
dover-lap $out_dir/all.rttm  $out_dir/* --label-mapping hungarian
# finally










