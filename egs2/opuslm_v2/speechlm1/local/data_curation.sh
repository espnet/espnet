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

# Input data specification
registered_data="\
  laion_audio_300m_part1 \
  laion_audio_300m_part2 \
  laion_audio_300m_part3 \
  laion_audio_300m_part4 \
  owsm_v4_caption \
  clotho_aqa \
  clotho_train \
  mtg-jamendo-dataset \
  emilia_en \
  laion_captioned_ai_music_snippets \
  laion_in_the_wild_sound_events \
  yt8m \
  yodas_auto \
  yodas_manual \
  audiocaps \
  audioset \
  fma \
  wavcaps \
  youtube_8m_arkive \
  laion_disco_12m_part1 \
  laion_disco_12m_part2 \
"
data_root=/work/nvme/bbjs/shared/opuslm_v2_data
stats_root=/work/nvme/bbjs/jtian1/espnet_speechlm_dev_sep/egs2/opuslm_v2/speechlm1/exp/stats_qwen3
minhush_root=/work/nvme/bbjs/shared/opuslm_v2_data/minhush/13_0.8
registry_file=/work/nvme/bbjs/shared/opuslm_v2_data/data_jsons/opuslm_v2.yaml
debug=false

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ -z "${registered_data}" ]; then
    log "Error: --registered_data is required"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Heuristic filtering"
    for dataset in ${registered_data}; do
        log "Processing dataset: ${dataset}"
        mkdir -p ${data_root}/data_curation/stage1_heuristic/${dataset}

        python3 local/heuristic_filtering.py \
            --input_dir ${data_root}/rich_caption/${dataset} \
            --output_dir ${data_root}/data_curation/stage1_heuristic/${dataset} \
            --num_workers 128
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Aggregate information"
    for dataset in ${registered_data}; do
        log "Processing dataset: ${dataset}"
        mkdir -p ${data_root}/data_curation/stage2_aggregate/${dataset}

        python3 local/aggregate_info.py \
            --audio_dir ${data_root}/audio/${dataset} \
            --rich_caption_dir ${data_root}/rich_caption/${dataset} \
            --heuristic_delete_ids ${data_root}/data_curation/stage1_heuristic/${dataset}/delete_ids.jsonl \
            --mos_dir ${data_root}/aes_clap/${dataset}/mos \
            --clap_dir ${data_root}/aes_clap/${dataset}/clap \
            --aesthetics_dir ${data_root}/aes_clap/${dataset}/aesthetics \
            --llm_judge_dir ${data_root}/llm_judge/${dataset} \
            --stats_text_to_audio ${stats_root}/stats_text_to_audio_${dataset}.jsonl \
            --stats_audio_to_text ${stats_root}/stats_audio_to_text_${dataset}.jsonl \
            --output_dir ${data_root}/data_curation/stage2_aggregate/${dataset} \
            --num_workers 128
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Analyze distribution"

    # Analyze music distribution
    python3 local/stage3_analyze_distribution.py \
        --datasets "${registered_data}" \
        --input_base_dir ${data_root}/data_curation/stage2_aggregate \
        --output_base_dir ${data_root}/data_curation/stage3_distribution_music \
        --minhush_root ${minhush_root} \
        --use_minhush \
        --is_pure_english true \
        --audio_type music \
        --num_workers 128 &
    
    python3 local/stage3_analyze_distribution.py \
        --datasets "${registered_data}" \
        --input_base_dir ${data_root}/data_curation/stage2_aggregate \
        --output_base_dir ${data_root}/data_curation/stage3_distribution_sound \
        --minhush_root ${minhush_root} \
        --use_minhush \
        --is_pure_english true \
        --audio_type sound_effects \
        --num_workers 128 &
    
    python3 local/stage3_analyze_distribution.py \
        --datasets "${registered_data}" \
        --input_base_dir ${data_root}/data_curation/stage2_aggregate \
        --output_base_dir ${data_root}/data_curation/stage3_distribution_speech \
        --minhush_root ${minhush_root} \
        --use_minhush \
        --is_pure_english true \
        --audio_type speech \
        --num_workers 128 &
    
    wait
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Filtering analysis with Gumbel top-k selection"

    # Understanding
    python3 local/stage4_filtering.py \
        --datasets "${registered_data}" \
        --audio_type music \
        --input_dir ${data_root}/data_curation/stage3_distribution_music \
        --output_dir ${data_root}/data_curation/stage4_filtering_music_und \
        --discard_ratio 0.20 \
        --temperature 0.3 \
        --num_workers 32 &

    python3 local/stage4_filtering.py \
        --datasets "${registered_data}" \
        --audio_type sound \
        --input_dir ${data_root}/data_curation/stage3_distribution_sound \
        --output_dir ${data_root}/data_curation/stage4_filtering_sound_und \
        --discard_ratio 0.20 \
        --temperature 0.3 \
        --num_workers 32 &

    python3 local/stage4_filtering.py \
        --datasets "${registered_data}" \
        --audio_type speech \
        --input_dir ${data_root}/data_curation/stage3_distribution_speech \
        --output_dir ${data_root}/data_curation/stage4_filtering_speech_und \
        --discard_ratio 0.42 \
        --temperature 0.3 \
        --num_workers 32 &

    # Generation
    python3 local/stage4_filtering.py \
        --datasets "${registered_data}" \
        --audio_type music \
        --input_dir ${data_root}/data_curation/stage3_distribution_music \
        --output_dir ${data_root}/data_curation/stage4_filtering_music_gen \
        --discard_ratio 0.20 \
        --temperature 0.1 \
        --num_workers 32 &

    python3 local/stage4_filtering.py \
        --datasets "${registered_data}" \
        --audio_type sound \
        --input_dir ${data_root}/data_curation/stage3_distribution_sound \
        --output_dir ${data_root}/data_curation/stage4_filtering_sound_gen \
        --discard_ratio 0.20 \
        --temperature 0.1 \
        --num_workers 32 &

    python3 local/stage4_filtering.py \
        --datasets "${registered_data}" \
        --audio_type speech \
        --input_dir ${data_root}/data_curation/stage3_distribution_speech \
        --output_dir ${data_root}/data_curation/stage4_filtering_speech_gen \
        --discard_ratio 0.28 \
        --temperature 0.1 \
        --num_workers 32 &
    
    wait
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Build curated datasets"

    for audio_type in music ; do
       
        python3 local/stage5_build_curated_dataset.py \
            --datasets "${registered_data}" \
            --audio_types ${audio_type} \
            --registry_file ${registry_file} \
            --stage4_root ${data_root}/data_curation \
            --stats_root ${stats_root} \
            --output_root ${data_root}/data_curation \
            --version und \
            --num_workers 32 &
        
        python3 local/stage5_build_curated_dataset.py \
            --datasets "${registered_data}" \
            --audio_types ${audio_type} \
            --registry_file ${registry_file} \
            --stage4_root ${data_root}/data_curation \
            --stats_root ${stats_root} \
            --output_root ${data_root}/data_curation \
            --version gen \
            --num_workers 32 &
        
        break
    done; wait
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Compute length statistics"

    python3 local/compute_length_stats.py \
        --datasets "${registered_data}" \
        --audio_types "music sound speech" \
        --stats_root ${stats_root} \
        --version und \
        --num_workers 32
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
