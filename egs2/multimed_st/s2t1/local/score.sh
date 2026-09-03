#!/usr/bin/env bash
set -euo pipefail

test_sets=

. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 [--test_sets test] <s2t_exp>" >&2
    exit 1
fi

s2t_exp=$1

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -n "${test_sets}" ]; then
    hyp_files=()
    for dset in ${test_sets}; do
        while IFS= read -r -d '' hyp; do
            hyp_files+=("${hyp}")
        done < <(find "${s2t_exp}" -path "*/${dset}/text_nospecial" -print0)
    done
else
    hyp_files=()
    while IFS= read -r -d '' hyp; do
        hyp_files+=("${hyp}")
    done < <(find "${s2t_exp}" -name text_nospecial -print0)
fi

if [ ${#hyp_files[@]} -eq 0 ]; then
    log "No text_nospecial files found under ${s2t_exp}; skip ST BLEU scoring"
    exit 0
fi

for hyp in "${hyp_files[@]}"; do
    dir=$(dirname "${hyp}")
    dset=$(basename "${dir}")
    ref="dump/raw/${dset}/text"

    if [ ! -f "${ref}" ]; then
        log "Reference ${ref} not found for ${hyp}; skip"
        continue
    fi

    scoredir="${dir}/score_st_bleu"
    mkdir -p "${scoredir}"

    python local/align_ref_hyp.py \
        "${ref}" "${hyp}" "${scoredir}/ref.txt" "${scoredir}/hyp.txt"

    echo "Case-sensitive ST result (single-reference)" > "${scoredir}/result.txt"
    sacrebleu "${scoredir}/ref.txt" \
        -i "${scoredir}/hyp.txt" \
        -m bleu chrf ter \
        >> "${scoredir}/result.txt"

    log "Write ST BLEU/chrF/TER result in ${scoredir}/result.txt"
    cat "${scoredir}/result.txt"
done
