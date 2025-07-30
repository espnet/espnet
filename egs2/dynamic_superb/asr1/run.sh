#!/usr/bin/env bash
set -e
set -u
set -o pipefail

stage=1
stop_stage=100
ngpu=1

. ./path.sh
. ./cmd.sh

. utils/parse_options.sh

# Stage 1-5: Data preparation
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 1-5: Data preparation"
    ./local/data.sh
fi

# Stage 12: Inference using integrated Qwen2-Audio model
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "Stage 12: Qwen2-Audio Inference with ESPnet2 integration"

    for dset in ./data/*; do
        _dset=${dset}
        echo ${_dset}
        _dir="exp/qwen2audio_inference/$(basename ${_dset})"
        _logdir="${_dir}/log"

        mkdir -p "${_logdir}"

        ${cuda_cmd} --gpu "${ngpu}" "${_logdir}/inference.log" \
            python -m espnet2.bin.prompt_s2t_inference \
                --ngpu "${ngpu}" \
                --data_path_and_name_and_type "${_dset}/wav.scp,speech,sound" \
                --data_path_and_name_and_type "${_dset}/text.input,text,text" \
                --output_dir "${_dir}" \
                --batch_size 1 \
                --decode_config_path "conf/decode_asr.yaml"
    done
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
  echo "Stage 13: Scoring with score_sclite.sh"
  for dset in ./data/*; do
    _dir="exp/qwen2audio_inference/$(basename ${dset})"
    ref=${dset}/text.output
    hyp=${_dir}/text

    mkdir -p ${_dir}/score

    # Text normalization function
    normalize_text() {
      local input_file=$1
      local output_file=$2

      # Apply text normalization: lowercase, remove punctuation, normalize whitespace
      awk '{
        # Extract utterance ID and text
        utt = $1
        $1 = ""
        text = substr($0, 2)

        # Convert to lowercase
        text = tolower(text)

        # Remove punctuation (keep only alphanumeric and spaces)
        gsub(/[^a-z0-9 ]/, "", text)

        # Normalize multiple spaces to single space
        gsub(/[ \t]+/, " ", text)

        # Remove leading/trailing whitespace
        gsub(/^[ \t]+|[ \t]+$/, "", text)

        # Print normalized text with utterance ID
        print utt " " text
      }' ${input_file} > ${output_file}
    }

    # Apply text normalization
    echo "Normalizing reference and hypothesis texts..."
    normalize_text ${ref} ${_dir}/score/ref.normalized
    normalize_text ${hyp} ${_dir}/score/hyp.normalized

    # Word Error Rate (using normalized texts)
    # strip utterance IDs, keep only the text
    awk '{ utt=$1; $1=""; print substr($0,2) " (spk-" utt ")" }' ${_dir}/score/ref.normalized > ${_dir}/score/ref.txt
    awk '{ utt=$1; $1=""; print substr($0,2) " (spk-" utt ")" }' ${_dir}/score/hyp.normalized > ${_dir}/score/hyp.txt

    # run sclite
    sclite \
        -r ${_dir}/score/ref.txt trn \
        -h ${_dir}/score/hyp.txt trn \
        -i rm -o sum stdout \
        -F \
        | tee ${_dir}/score/wer_sclite.log

    echo "WER report saved to ${_dir}/score/wer_sclite.log"

    # Character Error Rate (using normalized texts)
    # 1) Convert each line into "one char per token" plus the utt-ID:
    awk '{ utt=$1; $1=""; txt=substr($0,2); gsub(/ /,"",txt);
       spaced=""; for(i=1;i<=length(txt);i++){spaced=spaced substr(txt,i,1)" ";}
       print spaced "(spk-" utt ")" }' ${_dir}/score/ref.normalized \
        > ${_dir}/score/ref.char.trn

    awk '{ utt=$1; $1=""; txt=substr($0,2); gsub(/ /,"",txt);
       spaced=""; for(i=1;i<=length(txt);i++){spaced=spaced substr(txt,i,1)" ";}
       print spaced "(spk-" utt ")" }' ${_dir}/score/hyp.normalized \
        > ${_dir}/score/hyp.char.trn

    # 2) Run sclite in transcription mode ("trn") but it's now character‐level:
    sclite \
        -r ${_dir}/score/ref.char.trn trn \
        -h ${_dir}/score/hyp.char.trn trn \
        -i rm \
        -o sum stdout \
        | tee ${_dir}/score/cer_sclite.log
    echo "CER report saved to ${_dir}/score/cer_sclite.log"
  done
fi
