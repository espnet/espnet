#!/usr/bin/env bash

echo "$0 $*" >&2 # Print the command line for logging

nj=10
cmd=run.pl
evaltypes="SDR STOI ESTOI PESQ"
permutation=true
# Use museval.metrics.bss_eval_images or museval.metrics.bss_eval_source
bss_eval_images=true

. ./path.sh
. utils/parse_options.sh

if [ $# != 3 ]; then
    cat << EOF 1>&2
Usage: $0 reffiles enffiles <dir>
    e.g. $0 reference.scp enhanced.scp outdir

And also supporting multiple sources:
    e.g. $0 "ref1.scp,ref2.scp" "enh1.scp,enh2.scp" outdir

Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
EOF
    exit 1;
fi

set -euo pipefail

reffiles=( $(echo $1 | tr , " ") )
enhfiles=( $(echo $2 | tr , " ") )
dir=$3
logdir=${dir}/log
mkdir -p ${logdir}

split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/key.${n}.scp"
done

# Split the first reference
utils/split_scp.pl ${reffiles[0]} ${split_scps} || exit 1;

${cmd} JOB=1:${nj} ${logdir}/eval-enhanced-speech.JOB.log \
    eval-source-separation.py \
    --ref "${reffiles[@]}" --enh "${enhfiles[@]}" \
    --keylist ${logdir}/key.JOB.scp \
    --out ${logdir}/JOB \
    --evaltypes ${evaltypes} \
    --permutation ${permutation} \
    --bss_eval_images ${bss_eval_images}


for t in ${evaltypes/SDR/SDR ISR SIR SAR}; do
    for i in $(seq 1 ${nj}); do
        cat ${logdir}/${i}/${t}
    done > ${dir}/${t}

    # Calculate the mean over files
    python << EOF > ${dir}/mean_${t}
with open('${dir}/${t}', 'r') as f:
    values = []
    for l in f:
        vs = l.rstrip().split(None)[1:]
        values.append(sum(map(float, vs)) / len(vs))
    mean = sum(values) / len(values)
print(mean)
EOF

done
