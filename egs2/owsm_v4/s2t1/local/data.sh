#!/usr/bin/env bash
# This script prepares training data by filtering the YODAS2 dataset.
# See Figure 1 and Section 2.1 in the paper https://arxiv.org/pdf/2506.00338

set -e
set -u
set -o pipefail

min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}


stage=1
stop_stage=100
nj=4000

file_list= # Paths to the json files to be processed
yodas2_data_path= # Path to the YODAS2 dataset directory
save_dir= # Directory to save the intermediate data

. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

longform_data_file="${save_dir}/all_data.jsonl"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Section 2.1.1 in the paper:
    # YODAS provides unsegmented long-form recordings, but the timestamps can
    # be inaccurate. Our first step is to realign the audio and text using
    # the CTC segmentation algorithm.
    echo "Stage 1: Resegmentation"
    if [ -z "${file_list}" ]; then
        echo "Please specify the file_list variable with paths to resegmented data files."
        exit 1
    fi
    _logdir=logdir/ctc_seg
    mkdir -p ${_logdir}
    key_file=${file_list}

    _nj=$(min "${nj}" "$(wc -l ${key_file})")
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/data.${n}"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}

    # Perform CTC segmentation
    slurm.pl --gpu 1 --time 2:00:00 JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        python local/ctc_seg.py --file_list ${_logdir}/data.JOB

    # Create long-form utterances by concatenating consecutive segments
    python local/get_longform_from_reseg.py --yodas2_dir "${yodas2_data_path}" \
        --output_file "${longform_data_file}"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Section 2.1.2 in the paper:
    # Some utterances have incorrect language labels. To address this issue,
    # we perform LID on both the audio and text using public models.
    echo "Stage 2: LID-based filtering"
    _logdir=logdir/lid_reseg
    mkdir -p ${_logdir}
    key_file="${longform_data_file}"
    if [ ! -f "${key_file}" ]; then
        echo "Key file ${key_file} does not exist. Please run stage 1 first."
        exit 1
    fi

    _nj=$(min "${nj}" "$(wc -l ${key_file})")
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/data.${n}"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}

    # Perform LID on audio and text transcriptions
    slurm.pl --gpu 1 --time 1:00:00 JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        python local/lid.py --in_file ${_logdir}/data.JOB --out_file ${_logdir}/lid.JOB

    cat ${_logdir}/lid.* > ${save_dir}/lid.jsonl

    # Filter out utterances with mismatched language labels
    python local/filter_lid.py --in_file ${save_dir}/lid.jsonl \
        --out_file ${save_dir}/lid_remaining.txt
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Section 2.1.3 in the paper:
    # We filter out utterances with bad audio-text alignments,
    # as indicated by the CTC score computed in CTC segmentation.
    echo "Stage 3: CTC-score-based filtering"
    python local/filter_score.py --in_file ${save_dir}/lid_remaining.txt \
        --all_data ${longform_data_file} \
        --out_dir ${save_dir}/data
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # Convert the filtered data to Kaldi format for later training
    echo "Stage 4: Convert to Kaldi format"
    python local/convert_to_kaldi.py --in_dir ${save_dir}/data \
        --out_dir data/yodas0.10
    echo "Data conversion completed. Output saved to data/yodas0.10."
    echo "You can now use the data in Kaldi format for further processing."
fi
