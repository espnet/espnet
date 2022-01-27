#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# This script computes CER of Mandarin and WER of English separately

if [ $# -eq 1 ]; then
    exp=$1
else
    echo "only one argument is required"
fi

while IFS= read -r expdir; do
    if ls "${expdir}"/*/*/score_wer/hyp.trn &> /dev/null; then
        for scoredir in "${expdir}"/*/*/score_wer; do
            # split Mandarin and English transcriptions
            local/split_lang_trn.py -t ${scoredir}/hyp.trn -o ${scoredir}
            local/split_lang_trn.py -t ${scoredir}/ref.trn -o ${scoredir}

            # respectively computes the error rates
            for lang in eng man; do
                sclite -e utf-8 -c NOASCII \
                    -r "${scoredir}/ref.trn.${lang}" trn \
                    -h "${scoredir}/hyp.trn.${lang}" trn \
                    -i rm -o all stdout \
                    > "${scoredir}/result.${lang}.txt"
            done
        done

        # show results
        for lang in eng man; do
            if [ $lang = eng ]; then
                echo "English WER"
            else
                echo "Mandarin CER"
            fi

            echo "|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|"
            grep -H -e Avg "${expdir}"/*/*/score_wer/result.${lang}.txt \
                | sed -e "s#${expdir}/\([^/]*/[^/]*\)/score_wer/result.${lang}.txt:#|\1#g" \
                | sed -e 's#Sum/Avg##g' | tr '|' ' ' | tr -s ' ' '|'
            echo
        done
    fi
done < <(find ${exp} -mindepth 0 -maxdepth 1 -type d)
