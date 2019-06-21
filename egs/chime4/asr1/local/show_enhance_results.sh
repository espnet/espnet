#!/bin/bash
set -euo pipefail

. ./utils/parse_options.sh

if [ $# -ne 1 ]; then
    echo "$0 --recog-set <prefix>"
    exit 1
fi

prefix=$1
tasks=$(for i in $(ls -d ${prefix}*); do echo ${i} | sed -e "s#^${prefix}##"; done)

for etype in SDR ISR SIR SAR STOI ESTOI PESQ; do
    echo "### ${etype}"
    echo
    echo "|dataset|PED|CAF|STR|BUS|MEAN|"
    echo "|---|---|---|---|---|---|"
    for rtask in ${tasks}; do
        if [ -e ${prefix}${rtask}/eval_PED/mean_${etype} ]; then
            val=""
            for place in PED CAF STR BUS; do
                val+="$(printf %.4g $(cat ${prefix}${rtask}/eval_${place}/mean_${etype}))|"
            done
            val+=$(for place in PED CAF STR BUS; do
                cat ${prefix}${rtask}/eval_${place}/mean_${etype}
            done | awk 'BEGIN{sum=0;}{sum+=$1;}END{ print sum/NR "|"; }')
            echo "|${rtask}|${val}"
        fi
    done
done
