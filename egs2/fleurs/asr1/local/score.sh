#!/usr/bin/env bash
score_lang_id=true
decode_folder=
. utils/parse_options.sh

if [ $# -gt 1 ]; then
    echo "Usage: $0 --score_lang_id ture [exp]" 1>&2
    echo ""
    echo "Language Identification Scoring."
    echo 'The default of <exp> is "exp/".'
    exit 1
fi

[ -f ./path.sh ] && . ./path.sh
set -euo pipefail
if [ $# -eq 1 ]; then
    exp=$1
else
    exp=exp
fi

if [ "${score_lang_id}" = false ]; then
    echo "Training without language id, skip language identification scoring"
    exit 1
fi

echo "Preparing model output for language identification scoring"

python local/score_lang_id.py --exp_folder $exp --decode_folder $decode_folder