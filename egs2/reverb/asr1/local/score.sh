#!/usr/bin/env bash
set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0
help_message=$(cat << EOF
Usage: $0 <expdir>
EOF
)


log "$0 $*"
. ./utils/parse_options.sh
. ./path.sh


if [ $# -ne 1 ]; then
  log "${help_message}"
  exit 2
fi


exp=$1
for d in "${exp}"/*/*; do
    for e in wer cer ter; do
        if [ -e "${d}/score_${e}"/hyp.trn ] && [ -e "${d}/score_${e}"/ref.trn  ]; then
            for fn in far near; do
                if <"${d}/score_${e}"/hyp.trn grep -e "for_[0-9]ch_${fn}" 2>&1 >/dev/null; then
                    if [[ "${d}" =~ .*${fn} ]]; then
                        continue
                    fi
                    mkdir -p "${d}_${fn}/score_${e}"
                    <"${d}/score_${e}/hyp.trn" grep -e "for_[0-9]ch_${fn}" > "${d}_${fn}/score_${e}/hyp.trn"
                    <"${d}/score_${e}/ref.trn" grep -e "for_[0-9]ch_${fn}" > "${d}_${fn}/score_${e}/ref.trn"

                    sclite \
                        -r "${d}_${fn}/score_${e}/ref.trn" trn \
                        -h "${d}_${fn}/score_${e}/hyp.trn" trn \
                        -i rm -o all stdout > "${d}_${fn}/score_${e}/result.txt"

                    log "Write ${e} result in ${d}_${fn}/score_${e}/result.txt"
                    grep -e Avg -e SPKR -m 2 "${d}_${fn}/score_${e}/result.txt"
                fi
            done
        fi
    done
done

log "Successfully finished. [elapsed=${SECONDS}s]"
