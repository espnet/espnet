#!/usr/bin/env bash
# Generate MFA alignement
# You need to install the following tools to run this script:
# $ conda config --append channels conda-forge
# $ conda install montreal-forced-aligner
# If you are not using LJSpeech, be sure to define how your
# dataset is processed in `scripts/utils/mfa_format.py`.
set -e

if [[ "$(basename "$(pwd)")" != tts* ]]; then
  echo "You must cd to a tts directory"
  exit
fi

acoustic_model="english_mfa"
dictionary="english_us_mfa"
g2p_model="english_us_mfa"
dataset="ljspeech"
wavs_dir="downloads/LJSpeech-1.1/wavs"

. utils/parse_options.sh

mfa models download acoustic "$acoustic_model"
mfa models download dictionary "$dictionary"
mfa models download g2p "$g2p_model"
python scripts/utils/mfa_format.py labs --dataset "$dataset" --wavs_dir "$wavs_dir"

set +e

mfa validate "$wavs_dir" "$dictionary" "$acoustic_model" --brackets '' | while read -r line; do
  if [[ $line =~ "jobs" ]]; then
    echo "OOV file created, stopping MFA"
    pkill mfa
  else
    echo "$line"
  fi
done

set -e

mfa_dir="$HOME/Documents/MFA"
dict_dir="$mfa_dir/pretrained_models/dictionary"
oov_dict="$dict_dir/oov_$dataset.dict"
mfa g2p "$g2p_model" "$mfa_dir/wavs_validate_pretrained/oovs_found_$dictionary.txt" "$oov_dict"
cat "$dict_dir/$dictionary.dict" "$oov_dict" > "$dict_dir/${dictionary}_$dataset.dict"
mfa validate "$wavs_dir" "${dictionary}_$dataset" "$acoustic_model" --brackets ''
mfa align "$wavs_dir" "${dictionary}_$dataset" "$acoustic_model" ./textgrids

echo "Successfully finished generating MFA alignments."

# NOTE(iamanigeeit): If you want to train FastSpeech2 with the alignments,
# please check `egs2/ljspeech/tts1/run_mfa.sh`. For example:

#./run_mfa.sh --stage 0 --stop_stage 0
#./run_mfa.sh --stage 1 --stop_stage 1
#./run_mfa.sh --stage 2 --stop_stage 2
#./run_mfa.sh --stage 3 --stop_stage 3
#./run_mfa.sh --stage 4 --stop_stage 4
#./run_mfa.sh --stage 5 --stop_stage 5 \
#    --train_config conf/tuning/train_fastspeech2.yaml \
#    --teacher_dumpdir data \
#    --tts_stats_dir data/stats \
#    --write_collected_feats true
#./run_mfa.sh --stage 6 --stop_stage 6 \
#    --train_config conf/tuning/train_fastspeech2.yaml \
#    --teacher_dumpdir data \
#    --tts_stats_dir data/stats



