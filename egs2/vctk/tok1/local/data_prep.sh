#!/usr/bin/env bash

set -e
set -u
set -o pipefail

num_dev=5
num_eval=5
train_set=tr_no_dev
dev_set=dev
eval_set=eval1
microphone=mic1

. utils/parse_options.sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 [options] <VCTK-Corpus-0.92>" >&2
    exit 2
fi

corpus_dir=$1
text_root=${corpus_dir}/txt
if [ -d "${corpus_dir}/wav48_silence_trimmed" ]; then
    corpus_layout=v0.92
    audio_root=${corpus_dir}/wav48_silence_trimmed
    audio_pattern="*_${microphone}.flac"
elif [ -d "${corpus_dir}/wav48" ]; then
    corpus_layout=legacy
    audio_root=${corpus_dir}/wav48
    audio_pattern="*.wav"
else
    echo "Expected wav48_silence_trimmed or wav48 under ${corpus_dir}" >&2
    exit 1
fi
if [ ! -d "${text_root}" ]; then
    echo "Expected transcript directory ${text_root}" >&2
    exit 1
fi
if [ ! -r "${text_root}" ] || [ ! -x "${text_root}" ]; then
    echo "Transcript directory is not readable/searchable: ${text_root}" >&2
    exit 1
fi
if [ "${microphone}" != mic1 ] && [ "${microphone}" != mic2 ]; then
    echo "--microphone must be mic1 or mic2: ${microphone}" >&2
    exit 2
fi

train_dirs=()
dev_dirs=()
eval_dirs=()
temporary_dirs=()

for speaker_dir in "${audio_root}"/p*; do
    [ -d "${speaker_dir}" ] || continue
    speaker=$(basename "${speaker_dir}")
    # p315 has no transcripts in the official release.
    [ "${speaker}" = p315 ] && continue
    [ -d "${text_root}/${speaker}" ] || continue

    all_dir=data/${speaker}_all
    deveval_dir=data/${speaker}_deveval
    speaker_train_dir=data/${speaker}_${train_set}
    speaker_dev_dir=data/${speaker}_${dev_set}
    speaker_eval_dir=data/${speaker}_${eval_set}
    mkdir -p "${all_dir}"
    : > "${all_dir}/wav.scp"
    : > "${all_dir}/utt2spk"
    : > "${all_dir}/text"

    while IFS= read -r audio; do
        filename=$(basename "${audio}")
        if [ "${corpus_layout}" = v0.92 ]; then
            utterance=${filename%_${microphone}.flac}
        else
            utterance=${filename%.wav}
        fi
        transcript=${text_root}/${speaker}/${utterance}.txt
        [ -f "${transcript}" ] || continue
        text=$(tr '\r\n' '  ' < "${transcript}" | sed -e 's/[[:space:]]\+/ /g' -e 's/^ //' -e 's/ $//')
        [ -n "${text}" ] || continue
        echo "${utterance} ${audio}" >> "${all_dir}/wav.scp"
        echo "${utterance} ${speaker}" >> "${all_dir}/utt2spk"
        echo "${utterance} ${text}" >> "${all_dir}/text"
    done < <(find "${speaker_dir}" -maxdepth 1 -type f \
        -name "${audio_pattern}" | sort)

    utils/utt2spk_to_spk2utt.pl \
        "${all_dir}/utt2spk" > "${all_dir}/spk2utt"
    utils/fix_data_dir.sh "${all_dir}"
    num_all=$(wc -l < "${all_dir}/wav.scp")
    num_deveval=$((num_dev + num_eval))
    if [ "${num_all}" -le "${num_deveval}" ]; then
        echo "Too few usable utterances for ${speaker}: ${num_all}" >&2
        exit 1
    fi
    num_train=$((num_all - num_deveval))

    utils/subset_data_dir.sh --last \
        "${all_dir}" "${num_deveval}" "${deveval_dir}"
    utils/subset_data_dir.sh --first \
        "${deveval_dir}" "${num_dev}" "${speaker_dev_dir}"
    utils/subset_data_dir.sh --last \
        "${deveval_dir}" "${num_eval}" "${speaker_eval_dir}"
    utils/subset_data_dir.sh --first \
        "${all_dir}" "${num_train}" "${speaker_train_dir}"

    train_dirs+=("${speaker_train_dir}")
    dev_dirs+=("${speaker_dev_dir}")
    eval_dirs+=("${speaker_eval_dir}")
    # Keep the recipe data directory compact: these per-speaker directories
    # are only intermediates for the final speaker-closed split.
    temporary_dirs+=(
        "${all_dir}"
        "${deveval_dir}"
        "${speaker_train_dir}"
        "${speaker_dev_dir}"
        "${speaker_eval_dir}"
    )
done

if [ ${#train_dirs[@]} -eq 0 ]; then
    echo "No ${microphone} utterances found under ${audio_root}" >&2
    exit 1
fi

utils/combine_data.sh "data/${train_set}" "${train_dirs[@]}"
utils/combine_data.sh "data/${dev_set}" "${dev_dirs[@]}"
utils/combine_data.sh "data/${eval_set}" "${eval_dirs[@]}"

for temporary_dir in "${temporary_dirs[@]}"; do
    rm -rf "${temporary_dir}"
done

if [ "${corpus_layout}" = v0.92 ]; then
    echo "Prepared ${#train_dirs[@]} speakers from VCTK 0.92 using ${microphone}."
else
    echo "Prepared ${#train_dirs[@]} speakers from legacy VCTK wav48 data."
fi
