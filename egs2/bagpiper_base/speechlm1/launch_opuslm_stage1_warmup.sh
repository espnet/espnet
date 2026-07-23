#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=100

num_nodes=1
num_proc_per_node=4
node_rank=0
master_addr=localhost
master_port=8888


######## 1. Audio Training Data ########
datasets="\
  owsm_v4_caption \
  laion_audio_300m_part1 \
  laion_audio_300m_part2 \
  laion_audio_300m_part3 \
  laion_audio_300m_part4 \
  clotho_aqa \
  clotho_train \
  mtg-jamendo-dataset \
  emilia_en \
  laion_captioned_ai_music_snippets \
  laion_in_the_wild_sound_events \
  yt8m \
  youtube_8m_arkive \
  laion_disco_12m_part1 \
  laion_disco_12m_part2 \
  yodas_auto \
  yodas_manual \
  audiocaps \
  audioset \
  fma \
  wavcaps \
"

train_registered_specifier=""

# Understanding
music_und_ratio=3.15
for dataset in ${datasets}; do
  train_registered_specifier+="audio_to_text:${dataset}_music_und:${music_und_ratio} "
done

sound_und_ratio=3.17
for dataset in ${datasets}; do
  train_registered_specifier+="audio_to_text:${dataset}_sound_und:${sound_und_ratio} "
done

speech_und_ratio=1.0
for dataset in ${datasets}; do
  train_registered_specifier+="audio_to_text:${dataset}_speech_und:${speech_und_ratio} "
done

# Generation
music_gen_ratio=4.21
for dataset in ${datasets}; do
  train_registered_specifier+="text_to_audio:${dataset}_music_gen:${music_gen_ratio} "
done

sound_gen_ratio=4.27
for dataset in ${datasets}; do
  train_registered_specifier+="text_to_audio:${dataset}_sound_gen:${sound_gen_ratio} "
done

speech_gen_ratio=1.0
for dataset in ${datasets}; do
  train_registered_specifier+="text_to_audio:${dataset}_speech_gen:${speech_gen_ratio} "
done

######## 2. Text-only Data ########
text_ratio=1.0
train_registered_specifier+="\
  text_only:dolma3_ingredient1-code-meta-reasoning:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_adult_content:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_art_and_design:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_crime_and_law:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_education_and_jobs:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_electronics_and_hardware:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_entertainment:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_fashion_and_beauty:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_finance_and_business:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_food_and_dining:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_games:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_health:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_history_and_geography:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_home_and_hobbies:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_industrial:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_literature:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_politics:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_religion:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_science_math_and_technology:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_social_life:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_software:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_software_development:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_sports_and_fitness:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_transportation:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_19_travel_and_tourism:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_adult_content:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_art_and_design:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_crime_and_law:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_education_and_jobs:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_electronics_and_hardware:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_entertainment:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_fashion_and_beauty:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_finance_and_business:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_food_and_dining:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_games:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_health:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_history_and_geography:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_home_and_hobbies:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_industrial:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_literature:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_politics:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_religion:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_science_math_and_technology:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_social_life:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_software:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_software_development:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_sports_and_fitness:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_transportation:${text_ratio} \
  text_only:dolma3_ingredient1-common_crawl-high-quality_20_travel_and_tourism:${text_ratio} \
  text_only:dolma3_ingredient1-cranecode:${text_ratio} \
  text_only:dolma3_ingredient1-cranemath:${text_ratio} \
  text_only:dolma3_ingredient1-dolmino-math:${text_ratio} \
  text_only:dolma3_ingredient1-dolmino_1-flan:${text_ratio} \
  text_only:dolma3_ingredient1-general_reasoning_mix:${text_ratio} \
  text_only:dolma3_ingredient1-math-meta-reasoning:${text_ratio} \
  text_only:dolma3_ingredient1-megamatt:${text_ratio} \
  text_only:dolma3_ingredient1-nemotron-synth-qa:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-art_design-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-art_design-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-crime_law-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-crime_law-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-education_jobs-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-education_jobs-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-entertainment-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-entertainment-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-finance_business-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-finance_business-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-hardware-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-hardware-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-health-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-health-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-history-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-history-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-home_hobbies-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-home_hobbies-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-industrial-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-industrial-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-literature-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-literature-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-politics-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-politics-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-religion-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-religion-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-science_tech-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-science_tech-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-software-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-software-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-software_dev-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-software_dev-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-sports_fitness-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-sports_fitness-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-transportation-2e12:${text_ratio} \
  text_only:dolma3_ingredient1-olmocr_science_pdfs-high_quality-transportation-2e13:${text_ratio} \
  text_only:dolma3_ingredient1-omr-rewrite-fullthoughts:${text_ratio} \
  text_only:dolma3_ingredient1-program_verifiable:${text_ratio} \
  text_only:dolma3_ingredient1-reddit_to_flashcards:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_C:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_CSharp:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_Cpp:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_Go:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_Java:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_JavaScript:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_Markdown:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_PHP:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_Python:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_Ruby:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_Rust:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_SQL:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_Shell:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_Swift:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_15_TypeScript:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_C:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_CSharp:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_Cpp:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_Go:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_Java:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_JavaScript:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_Markdown:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_PHP:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_Python:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_Ruby:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_Rust:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_SQL:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_Shell:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_Swift:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_16_TypeScript:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_C:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_CSharp:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_Cpp:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_Go:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_Java:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_JavaScript:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_Markdown:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_PHP:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_Python:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_Ruby:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_Rust:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_SQL:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_Shell:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_Swift:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_17_TypeScript:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_C:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_CSharp:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_Cpp:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_Go:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_Java:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_JavaScript:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_Markdown:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_PHP:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_Python:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_Ruby:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_Rust:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_SQL:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_Shell:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_Swift:${text_ratio} \
  text_only:dolma3_ingredient1-stack_edu-fim_vigintile_19_TypeScript:${text_ratio} \
  text_only:dolma3_ingredient1-stem-heavy-crawl:${text_ratio} \
  text_only:dolma3_ingredient1-tinymath-mind:${text_ratio} \
  text_only:dolma3_ingredient1-tinymath-pot:${text_ratio} \
  text_only:dolma3_ingredient1-tulu-3-sft:${text_ratio} \
  text_only:dolma3_ingredient1-wiki_to_rcqa-part1:${text_ratio} \
  text_only:dolma3_ingredient1-wiki_to_rcqa-part2:${text_ratio} \
  dialogue:llama_nemotron:${text_ratio} \
  dialogue:olmo3_think:${text_ratio} \
  dialogue:olmo3_instruct:${text_ratio} \
"

######## 3. Validation Data ########
valid_registered_specifier="\
  audio_to_text:mmau_test_speech:1.0 \
  audio_to_text:mmau_test_sound:1.0 \
  audio_to_text:mmau_test_music:1.0 \
  text_to_audio:mmau_test_speech:1.0 \
  text_to_audio:mmau_test_sound:1.0 \
  text_to_audio:mmau_test_music:1.0 \
"

train_config=conf/train_stage1_qwen3_base.yaml

stats_dir=exp/stats_qwen3
exp_dir=exp/opuslm_v2_stage1_warmup_base
mkdir -p ${exp_dir}

inference_config=conf/inference.yaml
inference_step=10000
inference_nj=1

. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  python ../../../espnet2/speechlm/bin/prepare_length_stats.py \
    --train-registered-specifier "${train_registered_specifier}" \
    --valid-registered-specifier "${valid_registered_specifier}" \
    --train-config ${train_config} \
    --output-dir ${stats_dir} \
    --num-workers 88
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Node rank: ${node_rank} launch"

  mkdir -p ${exp_dir}/logs
  timestamp=$(date +"%Y-%m-%d_%H_%M")
  torchrun \
    --nnodes=${num_nodes} \
    --node_rank=${node_rank} \
    --nproc_per_node=${num_proc_per_node} \
    --master_addr=${master_addr} \
    --master_port=${master_port} \
      ../../../espnet2/speechlm/bin/train.py \
      --train-registered-specifier "${train_registered_specifier}" \
      --valid-registered-specifier "${valid_registered_specifier}" \
      --train-config ${train_config} \
      --stats-dir ${stats_dir} \
      --output-dir ${exp_dir} \
      --save-loader-state \
      --wandb-mode online \
      > ${exp_dir}/logs/train_node${node_rank}_${timestamp}.log 2>&1 
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  inference_tag=$(basename "${inference_config%.*}")

  inference_dir=${exp_dir}/inference/${inference_tag}_step_${inference_step}
  mkdir -p ${inference_dir}

  inference_ckpt=${exp_dir}/checkpoints/step_${inference_step}/global_step${inference_step}/mp_rank_00_model_states.pt

  echo "Start model inference. Log at ${inference_dir}/logs/inference.*.log"
  ${cuda_cmd} JOB=1:${inference_nj} ${inference_dir}/logs/inference.JOB.log \
    ../../../espnet2/speechlm/bin/inference.py \
      --rank JOB --world-size ${inference_nj} \
      --train-config ${exp_dir}/train.yaml \
      --inference-config ${inference_config} \
      --model-checkpoint ${inference_ckpt} \
      --output-dir ${inference_dir} \
      --test-registered-specifier "${test_registered_specifier}" \
      --num-worker 1
fi