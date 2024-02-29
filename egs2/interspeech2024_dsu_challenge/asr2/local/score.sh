#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./utils/parse_options.sh || exit 1;


if [ $# -ne 1 ]; then
    echo "Usage:  <asr_exp> "
    echo "Generate the submission result for the challenge."
    exit 1
fi

asr_exp=$1

# LibriSpeech eval sets
if ls "${asr_exp}"/*/{dev_clean,dev_other,test_clean,test_other}/score_cer/result.txt &> /dev/null; then
    ls_result=$(grep -H -e Sum "${asr_exp}"/*/{dev_clean,dev_other,test_clean,test_other}/score_cer/result.txt | \
        grep -v -e Avg | \
        sed -e "s#${asr_exp}/\([^/]*/[^/]*\)/score_cer/result.txt:#|\1#g" | \
        sed -e 's#Sum##g' | tr '|' ' ' | tr -s ' ' '|')
    final_ls_result=$(echo "${ls_result}" | awk -F"|" '{
            n=split($2, lst, "/");
            nutt_sum[lst[1]]+=$3; nchar_sum[lst[1]]+=$4; nerr_sum[lst[1]]+=$9;
            if (lst[1] in dset) {dset[lst[1]]=dset[lst[1]]"#"lst[2]} else {dset[lst[1]]=lst[2]}
        } END {
            for (inf_conf in dset) {
                n=split(dset[inf_conf], lst, "#");
                if (n==4) {
                    printf("%s/EN_LibriSpeech %.2f\n", inf_conf, 100*nerr_sum[inf_conf]/nchar_sum[inf_conf])
                }
            }
        }')
    # echo "${final_ls_result}"

else
    echo "Error: no librispeech eval set results found in ${asr_exp}. Exit." && exit 1;
fi

# ML-SUPERB eval sets
if ls "${asr_exp}"/*/test_1h/score_cer/result.txt &> /dev/null; then
    ms_result=$(grep -H -e Sum "${asr_exp}"/*/test_1h/score_cer/result.txt | \
        grep -v -e Avg | \
        sed -e "s#${asr_exp}/\([^/]*/[^/]*\)/score_cer/result.txt:#|\1#g" | \
        sed -e 's#Sum##g' | tr '|' ' ' | tr -s ' ' '|')
    final_ms_result=$(echo "${ms_result}" | awk -F"|" '{
            n=split($2, lst, "/");
            nutt_sum[lst[1]]=$3; nchar_sum[lst[1]]=$4; nerr_sum[lst[1]]=$9;
            dset[lst[1]]=lst[2];
        } END {
            for (inf_conf in dset) {
                if (dset[inf_conf] == "test_1h") {
                    printf("%s/ML_SUPERB %.2f\n", inf_conf, 100*nerr_sum[inf_conf]/nchar_sum[inf_conf])
                }
            }
        }')
    # echo "${final_ms_result}"

else
    echo "Error: no ml_superb eval set results found in ${asr_exp}. Exit." && exit 1;
fi

if [[ -n ${final_ls_result} ]] && [[ -n ${final_ms_result} ]]; then

    echo "## ${asr_exp}"
    echo "${ls_result}"
    echo "${ms_result}"
    echo

    all_results=$(echo "${final_ls_result}"; echo "${final_ms_result}")
    echo "${all_results}"
    echo

    echo "${all_results}" | awk '{
        n=split($1, lst, "/");
        if (lst[2]=="EN_LibriSpeech") {results[lst[1]]["en"]=$2;}
        else if (lst[2]=="ML_SUPERB") {results[lst[1]]["ml"]=$2;}
        } END {
            for (inf_conf in results) {
                if (length(results[inf_conf])==2) {
                    print(inf_conf, results[inf_conf]["en"], results[inf_conf]["ml"])
                }
            }
        }' | while read line; do
            output_file=$(echo "${line}" | cut -d" " -f1)"_result.json"
            echo "Writing submission json file in ${asr_exp}/${output_file}"
            echo

            cat <<EOF > ${asr_exp}/${output_file}
{
    "config": {
        "model_dtype": "torch.float16",
        "model_name": "YOUR_MODEL_NAME"
    },
    "results": {
        "asr_eval1": {
            "EN_LibriSpeech": $(echo "${line}" | cut -d" " -f2)
        },
        "asr_eval2": {
            "ML_SUPERB": $(echo "${line}" | cut -d" " -f3)
        },
        "asr_eval3": {
            "Bitrate": XX.XX
        }
    }
}
EOF

    done

    echo "Please compute the Bitrate information manually by following the README.md."

fi
