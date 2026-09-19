#!/usr/bin/env bash
# Symmetric re-scoring, called automatically at the end of s2t.sh stage 13
# (`[ -f local/score.sh ] && local/score.sh ${local_score_opts} "${s2t_exp}"`).
#
# Why: stage 13 builds hyp.trn from the special-token-stripped hypothesis
# ("text_nospecial") but ref.trn from the raw reference text, which still
# contains the prompt tokens "<nah_xxx><asr><notimestamps>". That asymmetry
# inflates CER/WER. Here we rebuild each ref.trn from the reference with all
# "<...>" tokens removed (matching the hypothesis) and re-run sclite.
#
# Idempotent: stripping an already-stripped reference is a no-op, so it is safe
# to run this repeatedly.
set -euo pipefail

# s2t.sh passes optional opts then the experiment dir as the last argument.
s2t_exp="${!#}"
RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$RECIPE_DIR"
# shellcheck disable=SC1091
. ./path.sh

if ! command -v sclite >/dev/null 2>&1; then
    echo "local/score.sh: WARNING: sclite not found on PATH; skipping re-scoring." >&2
    exit 0
fi

bpemodel="data/token_list/bpe_unigram50000/bpe.model"

# token type per score dir
tok_type() {
    case "$1" in
        *score_cer) echo char ;;
        *score_wer) echo word ;;
        *score_ter) echo bpe ;;
        *)          echo "" ;;
    esac
}

find "${s2t_exp}" -type d -name 'score_*' 2>/dev/null | while read -r scoredir; do
    [ -f "${scoredir}/hyp.trn" ] || continue
    ttype=$(tok_type "${scoredir}")
    [ -n "${ttype}" ] || continue

    dset=$(basename "$(dirname "${scoredir}")")
    # reference text used by stage 13 lives under the dump dir
    data="dump/raw/${dset}"
    [ -f "${data}/text" ] && [ -f "${data}/utt2spk" ] || continue

    _opts="--token_type ${ttype}"
    [ "${ttype}" = bpe ] && _opts+=" --bpemodel ${bpemodel}"

    # rebuild a symmetric reference: drop every "<...>" token, then tokenize the
    # same way the hypothesis was tokenized.
    paste \
        <(sed 's/<[^>]*>//g' "${data}/text" \
            | python3 -m espnet2.bin.tokenize_text -f 2- --input - --output - \
                --cleaner none ${_opts}) \
        <(awk '{ print "(" $2 "-" $1 ")" }' "${data}/utt2spk") \
        > "${scoredir}/ref.trn"

    sclite -r "${scoredir}/ref.trn" trn -h "${scoredir}/hyp.trn" trn \
        -i rm -o all stdout > "${scoredir}/result.txt"
    echo "local/score.sh: re-scored ${scoredir}"
    grep -E "Sum/Avg|Percent Total Error" "${scoredir}/result.txt" | head -1 || true
done
