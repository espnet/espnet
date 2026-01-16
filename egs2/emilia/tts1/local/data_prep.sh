#!/usr/bin/env bash

# Emilia dataset is used as training and development set
# VCTK dataset is used as evaluation set

nj=32
num_dev=5000
num_eval=5
train_set="tr_no_dev"
dev_set="dev"
eval_set="eval"
lang="EN"

. utils/parse_options.sh || exit 1;

db_emilia=$1
db_vctk=$2

if [ $# != 2 ]; then
    echo "Usage: $0 [Options] <emilia-db> <vctk-db>"
    echo "e.g.: $0 downloads/emilia downloads/VCTK-Corpus"
    echo ""
    echo "Options:"
    echo "    --num_dev: number of development uttreances (default=${num_dev})."
    echo "    --num_eval: number of evaluation uttreances of each speaker (default=${num_eval})."
    echo "    --train_set: name of train set (default=${train_set})."
    echo "    --dev_set: name of dev set (default=${dev_set})."
    echo "    --eval_set: name of eval set (default=${eval_set})."
    echo "    --nj: number of parallel jobs (default=${nj})."
    echo "    --lang: language code of Emilia dataset (default=${lang})."
    exit 1
fi

set -euo pipefail

# -------- Function: Preprocess Emilia Subset --------
preprocess_emilia_subset() {
    subset=$1
    db_emilia=$2

    subset_dir="data/${subset}_train"
    complete_flag="${subset_dir}/.complete"

    if [ -e "${complete_flag}" ]; then
        echo "[${subset}] already processed. Skipping..."
        return
    fi

    mkdir -p "${subset_dir}"
    scp="${subset_dir}/wav.scp"
    utt2spk="${subset_dir}/utt2spk"
    text="${subset_dir}/text"

    rm -f "${scp}" "${utt2spk}" "${text}"

    echo "[${subset}] Processing..."

    # Iterate over mp3 files and use jq for JSON parsing
    while IFS= read -r wav; do
        id=$(basename "${wav}" .mp3)
        json="${db_emilia}/${subset}/${id}.json"

        # Parse JSON fields
        speaker=$(jq -r '.speaker' "${json}")
        txt=$(jq -r '.text' "${json}")

        printf "%s %s\n" "$id" "$wav" >>"${scp}"
        printf "%s %s\n" "$id" "$speaker" >>"${utt2spk}"
        printf "%s %s\n" "$id" "$txt" >>"${text}"
    done < <(find "${db_emilia}/${subset}" -type f -name "*.mp3" | sort)

    utils/utt2spk_to_spk2utt.pl "${utt2spk}" >"${subset_dir}/spk2utt"
    touch "${complete_flag}"
    echo "[${subset}] Done."
}
export -f preprocess_emilia_subset

# Preprocess emilia data
echo "Preprocessing Emilia dataset..."
emilia_subsets=($(find "${db_emilia}" -maxdepth 1 -name "${lang}-*" -exec basename {} \; | sort))

# Run in parallel (adjust nj to your CPU cores)
parallel -j ${nj} preprocess_emilia_subset {} "${db_emilia}" ::: "${emilia_subsets[@]}"

echo "Combining Emilia subsets..."
# combine all subset of emilia dataset
utils/combine_data.sh data/traindev $(for subset in "${emilia_subsets[@]}"; do echo "data/${subset}_train"; done)

# remove each subset directory
for subset in ${emilia_subsets[@]}; do
    rm -rf data/${subset}_train
done

# split train and dev set
num_all=$(wc -l < data/traindev/wav.scp)
num_train=$((num_all - num_dev))
utils/subset_data_dir.sh --last data/traindev ${num_dev} data/${dev_set}
utils/subset_data_dir.sh --first data/traindev ${num_train} data/${train_set}
# remove temporary directory
rm -rf data/traindev

echo "Successfully prepared emilia data."

echo "Preprocessing vctk dataset..."
# Preprocess vctk data (Based on the script from VCTK recipe)
spks=$(find "${db_vctk}/wav48" -maxdepth 1 -name "p*" -exec basename {} \; | sort | grep -v p315)
eval_data_dirs=""
for spk in ${spks}; do
    # check spk existence
    [ ! -e "${db_vctk}/lab/mono/${spk}" ] && \
        echo "${spk} does not exist." >&2 && exit 1;

    [ ! -e data/${spk}_train ] && mkdir -p data/${spk}_train

    # set filenames
    scp=data/${spk}_train/wav.scp
    utt2spk=data/${spk}_train/utt2spk
    text=data/${spk}_train/text
    segments=data/${spk}_train/segments
    spk2utt=data/${spk}_train/spk2utt

    # check file existence
    [ -e "${scp}" ] && rm "${scp}"
    [ -e "${utt2spk}" ] && rm "${utt2spk}"
    [ -e "${text}" ] && rm "${text}"
    [ -e "${spk2utt}" ] && rm "${spk2utt}"
    [ -e "${segments}" ] && rm "${segments}"

    # make scp, text, and segments
    find "${db_vctk}/wav48/${spk}" -follow -name "*.wav" | sort | while read -r wav; do
        id=$(basename "${wav}" | sed -e "s/\.[^\.]*$//g")
        lab=${db_vctk}/lab/mono/${spk}/${id}.lab
        txt=${db_vctk}/txt/${spk}/${id}.txt

        # check lab existence
        if [ ! -e "${lab}" ]; then
            echo "${id} does not have a label file. skipped."
            continue
        fi
        if [ ! -e "${txt}" ]; then
            echo "${id} does not have a text file. skipped."
            continue
        fi

        echo "${id} ${wav}" >> "${scp}"
        echo "${id} ${spk}" >> "${utt2spk}"
        echo "${id} $(cat ${txt})" >> "${text}"

        utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"

        # parse start and end time from HTS-style mono label
        idx=1
        while true; do
            next_idx=$((idx+1))
            next_symbol=$(sed -n "${next_idx}p" "${lab}" | awk '{print $3}')
            if [ "${next_symbol}" != "pau" ]; then
                start_nsec=$(sed -n "${idx}p" "${lab}" | awk '{print $2}')
                break
            fi
            idx=${next_idx}
        done
        idx=$(wc -l < "${lab}")
        while true; do
            prev_idx=$((idx-1))
            prev_symbol=$(sed -n "${prev_idx}p" "${lab}" | awk '{print $3}')
            if [ "${prev_symbol}" != "pau" ]; then
                end_nsec=$(sed -n "${idx}p" "${lab}" | awk '{print $1}')
                break
            fi
            idx=${prev_idx}
        done
        start_sec=$(echo "${start_nsec}*0.0000001" | bc | sed "s/^\./0./")
        end_sec=$(echo "${end_nsec}*0.0000001" | bc | sed "s/^\./0./")
        echo "${id} ${id} ${start_sec} ${end_sec}" >> "${segments}"
    done

    # split
    utils/subset_data_dir.sh --last "data/${spk}_train" "${num_eval}" "data/${spk}_${eval_set}"
    # remove tmp directories
    rm -rf "data/${spk}_train"

    eval_data_dirs+=" data/${spk}_${eval_set}"
done

utils/combine_data.sh data/${eval_set} ${eval_data_dirs}

# remove tmp directories
rm -rf data/p[0-9]*

echo "Successfully prepared data."
