#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
wer=false

. utils/parse_options.sh

if [ $# != 3 ]; then
    echo "Usage: $0 <jsons> <dict> <out-dir>";
    exit 1;
fi

jsons=$1
dic=$2
dir=$3

tasks="\
SimData_dt_.*_near_room1 SimData_dt_.*_far_room1 \
SimData_dt_.*_near_room2 SimData_dt_.*_far_room2 \
SimData_dt_.*_near_room3 SimData_dt_.*_far_room3 \
RealData_dt_.*_near_room1 RealData_dt_.*_far_room1 \
SimData_et_.*_near_room1 SimData_et_.*_far_room1 \
SimData_et_.*_near_room2 SimData_et_.*_far_room2 \
SimData_et_.*_near_room3 SimData_et_.*_far_room3 \
RealData_et_.*_near_room1 RealData_et_.*_far_room1"

for task in ${tasks}; do
    filename=`echo ${task} | sed -e "s/\.\*_//"`
    mkdir -p ${dir}/${filename}
    python local/filterjson.py -f ${task} ${jsons} > ${dir}/${filename}/data.1.json
    score_sclite.sh --wer ${wer} --nlsyms ${nlsyms} ${dir}/${filename} ${dic} 1> /dev/null 2> /dev/null
done

echo "Scoring for the REVERB challenge"
tasks_eval="\
SimData_et_.*_near_room1 SimData_et_.*_far_room1 \
SimData_et_.*_near_room2 SimData_et_.*_far_room2 \
SimData_et_.*_near_room3 SimData_et_.*_far_room3 \
RealData_et_.*_near_room1 RealData_et_.*_far_room1"
for task in ${tasks_eval}; do
    filename=`echo ${task} | sed -e "s/\.\*_//"`
    echo "${filename}:"
    grep -e Avg -e SPKR -m 2 ${dir}/${filename}/result.wrd.txt
done | sed -e 's/\s\s\+/\t/g' | tee ${dir}/result.txt
