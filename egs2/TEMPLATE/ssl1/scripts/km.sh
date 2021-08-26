#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=100
train_set=
dev_set=
test_set=
datadir=
kmrootdir=
dictdir=

nclusters=100
feature_type=mfcc

# Extract intermediate Hubert embedding from official hubert model:
hubert_url="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
hubert_dir_path="./downloads/hubert_pretrained_models/hubert_base_ls960.pt"

# Extract intermediate Hubert embedding from espnet-trained model:
# hubert_url="espnet"
# hubert_dir_path="" # Pretrained Hubert model dir contains 'valid.acc.best.pth' and 'config.yaml'

portion=0.1
nj=1
python=python3       # Specify python to execute espnet commands.

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0 <--nclusters:100> <--feature_type:mfcc>"
    exit 0
fi

km_path="${kmrootdir}/km_${train_set}_${feature_type}/km_${nclusters}clusters.mdl"
mkdir -p "$(dirname ${km_path})"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Learn K-means with ${feature_type} feature based on scikit-learn"

    ${python} pyscripts/sklearn_km.py \
              --feats-dir "${datadir}/${train_set}" \
              --km-path "${km_path}" \
              --n-cluster "${nclusters}" \
              --feature-type "${feature_type}" \
              --hubert-model-url "${hubert_url}" \
              --hubert-model-path "${hubert_dir_path}" \
              --nj ${nj} \
              --portion ${portion}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Generate K-means pseudo-labels"
    
    for task in ${train_set} ${dev_set} ${test_set}; do
        # move ${datadir}/${task}/ to new folders and rename ptext
        plabel_dir="${datadir}/${task}_${feature_type}_km${nclusters}"
        if [[ -d "${plabel_dir}" ]]; then
            echo "${plabel_dir} already exists, will remove it"
            rm -r ${plabel_dir}
        fi
        mkdir -p ${plabel_dir}
        cp -r ${datadir}/${task}/* ${plabel_dir}
        
        ${python} pyscripts/dump_km_label.py \
                  --km-path "${km_path}" \
                  --label-path "${plabel_dir}/text" \
                  --recog-set "${plabel_dir}" \
                  --feature "${feature_type}" \
                  --hurl "${hubert_url}" \
                  --hdir "${hubert_dir_path}" \
                  --nj ${nj}
        
        utils/fix_data_dir.sh ${plabel_dir}
    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Generate char-based fairseq style dictionary: <token> <count>"
    # generate dictionaries
    oov="<unk>"         # Out of vocabulary symbol.
    blank="<blank>"     # CTC blank symbol
    pad="<pad>"
    sos_eos="<sos/eos>" # sos and eos symbole
    
    mkdir -p ${dictdir}
    
    <${datadir}/${train_set}_${feature_type}_km${nclusters}/text cut -d" " -f2- | \
        awk '{for (i=1; i<=NF; i++) {count[$i]+=1}} END{for (k in count) {print(k, count[k])}}' | \
        sort -n -r -k 2 | \
        awk -v oov=${oov} -v blank=${blank} -v sos_eos=${sos_eos} -v pad=${pad} \
            'BEGIN{print(blank, 0); print(oov, 0); print(pad, 0)} {print($0)} END{print(sos_eos, 0)}' > ${dictdir}/dict.txt
    
    <${datadir}/${train_set}_${feature_type}_km${nclusters}/text cut -d" " -f2- | \
        awk '{for (i=1; i<=NF; i++) {count[$i]+=1}} END{for (k in count) {print(k, count[k])}}' | \
        sort -n -r -k 2  | \
        awk -v oov=${oov} -v blank=${blank} -v sos_eos=${sos_eos} -v pad=${pad} \
            'BEGIN{print(blank); print(oov)} {print($1)} END{print(sos_eos)}' > ${dictdir}/tokens.txt
    
    log "Successfully generate the ${dictdir}/{dict,tokens}.txt"
    
fi
