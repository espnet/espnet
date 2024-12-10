#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0

stage=-2
stop_stage=1
lang='EN'
nj=32

help_message=$(cat << EOF
Usage: $0 [--stage <stage>] [--stop-stage <stop_stage>] [--lang <lang>] [--nj <nj>]

Options:
    --stage        Starting stage (default: ${stage})
    --stop-stage   Ending stage (default: ${stop_stage})
    --lang         Language selection (default: ${lang})
    --nj          Number of parallel jobs (default: ${nj})
    --help        Show this help message
EOF
)

while [ $# -gt 0 ]; do
    case "$1" in
        --stage)
            stage="$2"
            shift 2
            ;;
        --stop-stage)
            stop_stage="$2"
            shift 2
            ;;
        --lang)
            lang="$2"
            shift 2
            ;;
        --nj)
            nj="$2"
            shift 2
            ;;
        --help)
            echo "${help_message}"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "${help_message}"
            exit 1
            ;;
    esac
done

log "Stage start from ${stage} to ${stop_stage}"

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${GENSHIN}" ]; then
   log "Fill the value of 'GENSHIN' in db.sh"
   exit 1
fi
db_root=${GENSHIN}

train_set=tr_no_dev
train_dev=dev
eval_set=eval1

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
    log "Stage -2: Data download"
    bash local/data_download.sh "${db_root}" "${lang}"
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "Stage -1: Preprocess Data"
    python local/data_process.py "${db_root}/Genshin_${lang}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Data preparation"
    
    mkdir -p data/train

    tmpdir=$(mktemp -d)
    trap 'rm -rf ${tmpdir}' EXIT
    
    wav_scp=${tmpdir}/wav.scp
    utt2spk=${tmpdir}/utt2spk
    text=${tmpdir}/text

    : > "${wav_scp}"
    : > "${utt2spk}"
    : > "${text}"
    
    log "Creating speaker list..."
    find -L "${db_root}/Genshin-${lang}" -mindepth 1 -maxdepth 1 -type d -print0 | \
        xargs -0 -I{} basename {} | sort > "${tmpdir}/speakers.list"
    
    log "Processing speakers..."
    total_speakers=$(wc -l < "${tmpdir}/speakers.list")
    log "Total number of speakers: ${total_speakers}"

    process_speaker() {
        local spk=$1
        local spk_normalized=$(echo "${spk}" | tr -s " " "_" | tr -c "a-zA-Z0-9_-" "_" | sed 's/_*$//')
        local spk_dir="${db_root}/Genshin-${lang}/${spk}"
        local wav_scp_spk="${tmpdir}/wav_${spk_normalized}.scp"
        local utt2spk_spk="${tmpdir}/utt2spk_${spk_normalized}"
        local text_spk="${tmpdir}/text_${spk_normalized}"
        
        : > "${wav_scp_spk}"
        : > "${utt2spk_spk}"
        : > "${text_spk}"

        log "Processing speaker ${spk}..."
        
        find "${spk_dir}" -name "*.wav" -print0 | while IFS= read -r -d $'\0' wav_file; do
            lab_file="${wav_file%.*}.lab"
            base_name=$(basename "${wav_file}" .wav | tr -s " " "_" | tr -c "a-zA-Z0-9_-" "_" | sed 's/_*$//')

            utt_id="${spk_normalized}-${base_name}"
            
            if [ -f "${lab_file}" ] && [ -s "${lab_file}" ]; then
                text_content=$(cat "${lab_file}" | tr -s " " | sed "s/^ *//;s/ *$//")
                
                if [ ! -z "${text_content}" ]; then
                    abs_wav_path=$(readlink -f "${wav_file}")
                    echo "${utt_id} ${abs_wav_path}" >> "${wav_scp_spk}"
                    echo "${utt_id} ${spk_normalized}" >> "${utt2spk_spk}"
                    echo "${utt_id} ${text_content}" >> "${text_spk}"
                fi
            fi
        done

        if [ -f "${wav_scp_spk}" ]; then
            cat "${wav_scp_spk}" >> "${tmpdir}/wav.scp"
        fi
        if [ -f "${utt2spk_spk}" ]; then
            cat "${utt2spk_spk}" >> "${tmpdir}/utt2spk"
        fi
        if [ -f "${text_spk}" ]; then
            cat "${text_spk}" >> "${tmpdir}/text"
        fi
    }
    export -f process_speaker
    export -f log
    export db_root lang tmpdir

    parallel --jobs "${nj}" --bar process_speaker :::: "${tmpdir}/speakers.list"

    for type in wav.scp text; do
        sort -k1,1V "${tmpdir}/${type}" > "data/train/${type}"
    done

    chmod +x local/sort_by_id.pl
    local/sort_by_id.pl "${tmpdir}/utt2spk" > "data/train/utt2spk"

    log "Generating spk2utt..."
    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt

    LC_ALL=C sort -k1,1 data/train/spk2utt > data/train/spk2utt.tmp
    mv data/train/spk2utt.tmp data/train/spk2utt

    for f in wav.scp utt2spk text spk2utt; do
        n=$(wc -l < data/train/$f)
        log "Number of lines in $f: $n"
    done  

    local/sort_by_id.pl "data/train/utt2spk" > "data/train/utt2spk.tmp"
    mv "data/train/utt2spk.tmp" "data/train/utt2spk"
    
    utils/fix_data_dir.sh data/train
    utils/validate_data_dir.sh --no-feats data/train || {
        log "Data validation failed"
        exit 1
    }

    rm -rf data/train/.backup
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Split data into training, development and evaluation sets"
    
    total_utts=$(wc -l < data/train/wav.scp)
    
    dev_eval_size=$(( total_utts / 10 ))
    dev_size=$(( dev_eval_size / 2 ))
    eval_size=$(( dev_eval_size - dev_size ))
    
    log "Total utterances: ${total_utts}"
    log "Dev set size: ${dev_size}"
    log "Eval set size: ${eval_size}"
    
    utils/subset_data_dir.sh data/train ${dev_eval_size} data/deveval
    
    utils/subset_data_dir.sh data/deveval ${eval_size} data/${eval_set}
    utils/subset_data_dir.sh data/deveval ${dev_size} data/${train_dev}
    
    n=$(( total_utts - dev_eval_size ))
    utils/subset_data_dir.sh data/train ${n} data/${train_set}
    
    for dset in ${train_set} ${train_dev} ${eval_set}; do
        utils/fix_data_dir.sh data/${dset}
        utils/validate_data_dir.sh --no-feats data/${dset}
        
        num_utts=$(wc -l < data/${dset}/wav.scp)
        num_spks=$(wc -l < data/${dset}/spk2utt)
        log "Successfully prepared ${dset} (${num_utts} utterances, ${num_spks} speakers)"
    done
fi

elapsed=$SECONDS
log "Data preparation finished in ${elapsed} seconds"
