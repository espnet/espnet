#!/usr/bin/env bash
# Check the alignment labels of stage 4 (local/data.sh) against Montreal Forced
# Aligner word times on LibriSpeech dev-clean and dev-other: read clips with real
# pauses inside them, which the meeting test sets (AMI) do not have.
#
#   local/check_labels_librispeech.sh [--stage 0] [--stop_stage 2] [--out exp/label_check_librispeech]
#
# Stages:
#   0  download the MFA LibriSpeech alignments (local/v2/downloads/librispeech_mfa.tsv:
#      Zenodo 10.5281/zenodo.2619474, CC BY 4.0) into ${VAD_CORPORA}/librispeech_mfa
#   1  align every utterance on its own (one clip per recording, no packing) with the
#      OWSM-CTC setup of stage 4; keep the CTC segmentation, token and torchaudio
#      forced-alignment word spans (local/v2/align_label_check.py align); one GPU
#      task (${cuda_cmd}). Also writes mfa.rttm: the MFA words with gaps shorter
#      than 0.2 s joined, as the test RTTMs.
#   2  score each label set, and the MFA words as the ceiling, against mfa.rttm over
#      a grid of start pad x end pad x in-clip min_pause (local/v2/align_label_check.py
#      score, one CPU task per label set, ${train_cmd})
#   3  clip mode on real packs (local/v2/check_clip_mode.py): 60 dev-clean utterances
#      written packed and one per recording, both labelled by align.py --clip_mode
#      (${clip_check_options}); the labels of every clip must match within a frame
# MFA is an aligner, not a human: its word times are the reference here only
# because no human word timing of LibriSpeech exists.
set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=0
stop_stage=2
# the model and settings of stage 4 (local/data.sh), clip mode with token timing
clip_check_options="--model_tag espnet/owsm_ctc_v4_1B --revision 6075bf3d42e6037d1ef7abd7c5558ade886e6eaa"
clip_check_options+=" --batch_seconds 3840 --dtype float32 --use_sdpa true --tf32 false"
clip_check_options+=" --min_word_score -15 --word_timing tokens --pad_start 0.05 --pad_end 0.4"
clip_check_options+=" --min_pause 0.5"
out=exp/label_check_librispeech
subsets="dev-clean dev-other"
starts="0 0.025 0.05 0.1"
ends="0 0.1 0.2 0.3 0.4"
min_pauses="0.2 0.3 0.5 0.75 1.0 1.5 inf"
curve_pads="0.05 0.4"   # the min_pause curve is printed at these start and end pads

log "$0 $*"
. utils/parse_options.sh
. ./path.sh
. ./db.sh
. ./cmd.sh

v2=local/v2
mfa_zip=${VAD_CORPORA}/librispeech_mfa/ARCHIVES/librispeech_alignments.zip
mkdir -p "${out}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: MFA LibriSpeech alignments"
    python3 ${v2}/download.py --root "${VAD_CORPORA}" --corpora librispeech_mfa --jobs 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: align ${subsets} utterance by utterance"
    # shellcheck disable=SC2086
    ${cuda_cmd} --gpu 1 "${out}/align.log" \
        python3 ${v2}/align_label_check.py align --set librispeech \
            --dump "${VAD_CORPORA}/librispeech/LibriSpeech" --subsets ${subsets} \
            --annotations "${mfa_zip}" --out "${out}/librispeech" \
            --batch_seconds 3840 --dtype float32 --use_sdpa true --tf32 false \
            --min_word_score -15 --workers 12
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: score against the MFA words"
    sets=(segmentation tokens forced_align xml_words)
    printf '%s\n' "${sets[@]}" > "${out}/label_sets.txt"
    # shellcheck disable=SC2086,SC2016
    ${train_cmd} JOB=1:${#sets[@]} "${out}/score.JOB.log" \
        python3 ${v2}/align_label_check.py score --out "${out}/librispeech" \
            --ref "${out}/librispeech/mfa.rttm" \
            --label_set "\$(sed -n JOBp ${out}/label_sets.txt)" \
            --starts ${starts} --ends ${ends} --min_pauses ${min_pauses} \
            --curve_pads ${curve_pads}
    for i in $(seq 1 ${#sets[@]}); do grep -vE "^#" "${out}/score.${i}.log"; done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: clip mode on real packs"
    # shellcheck disable=SC2086
    ${cuda_cmd} --gpu 1 "${out}/clip_mode_check.log" \
        python3 ${v2}/check_clip_mode.py \
            --librispeech "${VAD_CORPORA}/librispeech/LibriSpeech" \
            --out "${out}/clip_mode_check" -- ${clip_check_options}
    grep -E "utterances in|DIFFERS" "${out}/clip_mode_check.log"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
