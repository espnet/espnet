#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0
stage=1
stop_stage=100000
rootdir=/work/nvme/bbjs/shared/opuslm_v2_data

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Dump datasets for LAION-Audio-300M"
    for part in 2; do
        # Audio is pre-dumpped
        audio_dir=${rootdir}/audio/laion_audio_300m_part${part}
        
        # Dump text to Arkive
        text_dir=${rootdir}/rich_caption/laion_audio_300m_part${part}
        mkdir -p ${text_dir}/dump

        python3 local/dump_text.py \
          --input_dir ${text_dir} \
          --output_dir ${text_dir}/dump \
          --mode qwen_caption \
          --file_regex '^captions_rank.+\.jsonl$' \
          --num_workers 128
        
        # Build data json
        json_dir=${rootdir}/data_jsons/laion_audio_300m_part${part}
        mkdir -p ${json_dir}
        python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
            --triplets audio1,${audio_dir}/metadata.parquet,arkive_audio \
                       text1,${text_dir}/dump/metadata.parquet,arkive_text \
            --output_json ${json_dir}/caption.json 
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Prepare OWSM v4"
    audio_dir=${rootdir}/audio/owsm_v4

    # Build audio-to-rich-caption dataset json
    text_dir=${rootdir}/rich_caption/owsm_v4
    # python3 local/dump_text.py \
    #     --input_dir ${text_dir} \
    #     --output_dir ${text_dir}/dump \
    #     --mode qwen_caption \
    #     --file_regex '^captions_rank.+\.jsonl$' \
    #     --num_workers 128

    json_dir=${rootdir}/data_jsons/owsm_v4; mkdir -p ${json_dir}
    python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
        --triplets audio1,${audio_dir}/metadata.parquet,arkive_audio \
                   text1,${text_dir}/dump/metadata.parquet,arkive_text \
        --output_json ${json_dir}/caption.json

    # Build audio-to-text dataset json
    text_dir=${rootdir}/text/owsm_v4
    # python3 local/dump_text.py \
    #     --input_dir ${text_dir} \
    #     --output_dir ${text_dir}/dump \
    #     --mode kaldi \
    #     --file_regex '^text$' \
    #     --num_workers 128
    
    python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
        --triplets audio1,${audio_dir}/metadata.parquet,arkive_audio \
                   text1,${text_dir}/dump/metadata.parquet,arkive_text \
        --output_json ${json_dir}/asr.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Prepare dolma3"
    text_dir=${rootdir}/text/dolma3_dolmino_mix-100B-1125
    json_dir=${rootdir}/data_jsons/dolma3; mkdir -p ${json_dir}

    for part in `ls ${text_dir}/data`; do
        
        if [ -f ${json_dir}/${part}.json ]; then
            echo "Already have ${json_dir}/${part}.json. Skip processing it"
            continue
        fi

        echo "working on ${text_dir}/data/${part}"
        mkdir -p ${text_dir}/data/${part}/dump
        python3 local/dump_text.py \
            --input_dir ${text_dir}/data/${part} \
            --output_dir ${text_dir}/data/${part}/dump \
            --mode dolma3 \
            --file_regex '.*\.jsonl$' \
            --num_workers 128

        python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
            --triplets text1,${text_dir}/data/${part}/dump/metadata.parquet,arkive_text \
            --output_json ${json_dir}/${part}.json
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Prepare llama nemotron"
    text_dir=${rootdir}/text/llama_nemotron
    json_dir=${rootdir}/data_jsons/llama_nemotron; mkdir -p ${json_dir}

    python3 local/dump_text.py \
        --input_dir ${text_dir} \
        --output_dir ${text_dir}/dump \
        --mode llama_nemotron \
        --file_regex '.*\.jsonl$' \
        --num_workers 8
    
    python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
        --triplets dialogue,${text_dir}/dump/metadata.parquet,arkive_dialogue \
        --output_json ${json_dir}/text.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Prepare OLMo-3 SFT data"

    # allenai/Dolci-Instruct-SFT
    text_dir=${rootdir}/text/olmo3_instruct
    json_dir=${rootdir}/data_jsons/olmo3_instruct; mkdir -p ${json_dir}

    python3 local/dump_olmo3_sft.py \
        --output_dir ${text_dir}/dump \
        --dataset allenai/Dolci-Instruct-SFT
    
    python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
        --triplets dialogue,${text_dir}/dump/metadata.parquet,arkive_dialogue \
        --output_json ${json_dir}/text.json

    # allenai/Dolci-Think-SFT-32B
    text_dir=${rootdir}/text/olmo3_think
    json_dir=${rootdir}/data_jsons/olmo3_think; mkdir -p ${json_dir}

    python3 local/dump_olmo3_sft.py \
        --output_dir ${text_dir}/dump \
        --dataset allenai/Dolci-Think-SFT-32B
    
    python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
        --triplets dialogue,${text_dir}/dump/metadata.parquet,arkive_dialogue \
        --output_json ${json_dir}/text.json
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Prepare multiple audio caption dataset"

    names=""
    names+="clotho_test "
    names+="clotho_aqa "
    names+="clotho_train "
    names+="mtg-jamendo-dataset "
    names+="yt8m "
    names+="laion_captioned_ai_music_snippets "
    names+="laion_in_the_wild_sound_events "
    names+="emilia_en"
    names+="audioset "
    names+="audiocaps "
    names+="fma "
    names+="mmau_test_speech "
    names+="mmau_test_sound "
    names+="mmau_test_music "
    names+="clotho_train_development "
    names+="youtube_8m_arkive "
    names+="laion_disco_12m_part1 "
    names+="laion_disco_12m_part2 "

    for name in ${names}; do

        audio_dir=${rootdir}/audio/${name}

        # Build audio-to-rich-caption dataset json
        text_dir=${rootdir}/rich_caption/${name}; mkdir -p ${text_dir}
        python3 local/dump_text.py \
            --input_dir ${text_dir} \
            --output_dir ${text_dir}/dump \
            --mode qwen_caption \
            --file_regex '^captions_rank.+\.jsonl$' \
            --num_workers 32

        json_dir=${rootdir}/data_jsons/${name}; mkdir -p ${json_dir}
        python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
            --triplets audio1,${audio_dir}/metadata.parquet,arkive_audio \
                    text1,${text_dir}/dump/metadata.parquet,arkive_text \
            --output_json ${json_dir}/caption.json
    done

    # Audio data packed by Kaldiio
    names=""
    names+="yodas_auto "
    names+="yodas_manual "
    names+="wavcaps "

    for name in ${names}; do

        audio_dir=${rootdir}/audio/${name}

        Build audio-to-rich-caption dataset json
        text_dir=${rootdir}/rich_caption/${name}; mkdir -p ${text_dir}
        python3 local/dump_text.py \
            --input_dir ${text_dir} \
            --output_dir ${text_dir}/dump \
            --mode qwen_caption \
            --file_regex '^captions_rank.+\.jsonl$' \
            --num_workers 32

        json_dir=${rootdir}/data_jsons/${name}; mkdir -p ${json_dir}
        python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
            --triplets audio1,${audio_dir}/wav.scp,kaldi_audio \
                    text1,${text_dir}/dump/metadata.parquet,arkive_text \
            --output_json ${json_dir}/caption.json
    done
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
