#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 [--stage <stage>] [--stop_stage <stop_stage>]

  optional argument:
    [--stage]: 1 (default) or 2 or 3
    [--stop_stage]: 1 or 2 or 3 (default)
EOF
)


stage=1
stop_stage=3
log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation for each dataset"

    cwd="$(pwd)"

    cd ../../../egs2/vctk_noisy/enh1
    if [ -e "vctk_noisy.done" ]; then
        log "Skipping VoiceBank+DEMAND as it is already prepared"
    else
        ./run.sh --stage 1 --stop_stage 4 --max_wav_duration 50 --fs 48k \
            --use_noise_ref false --use_dereverb_ref false
        touch vctk_noisy.done
    fi

    cd ../../../egs2/dns_ins20/enh1
    if [ -e "dns_ins20.done" ]; then
        log "Skipping DNS1 as it is already prepared"
    else
        ./run.sh --stage 1 --stop_stage 4 --max_wav_duration 50 \
            --use_noise_ref false --use_dereverb_ref false
        touch dns_ins20.done
    fi

    cd ../../../egs2/chime4/enh1
    if [ -e "chime4.done" ]; then
        log "Skipping CHiME-4 as it is already prepared"
    else
        ./run.sh --stage 1 --stop_stage 4 --max_wav_duration 50 \
            --train_set "tr05_simu_isolated_6ch_track" \
            --valid_set "dt05_simu_isolated_6ch_track" \
            --test_sets "et05_simu_isolated_6ch_track dt05_real_isolated_6ch_track et05_real_isolated_6ch_track" \
            --use_noise_ref false \
            --use_dereverb_ref false
        touch chime4.done
    fi

    cd ../../../egs2/reverb/enh1
    if [ -e "reverb.done" ]; then
        log "Skipping REVERB as it is already prepared"
    else
        ./run.sh --stage 1 --stop_stage 4 --max_wav_duration 50 \
            --use_noise_ref false --use_dereverb_ref false
        touch reverb.done
    fi

    cd ../../../egs2/whamr/enh1
    if [ -e "whamr.done" ]; then
        log "Skipping WHAMR! as it is already prepared"
    else
        ./run.sh --stage 1 --stop_stage 4 --fs 16k --max_wav_duration 50 \
            --train_set "tr_mix_single_reverb_max_16k" \
            --valid_set "cv_mix_single_reverb_max_16k" \
            --test_sets "tt_mix_single_reverb_max_16k" \
            --local_data_opts "--sample_rate 16k --min_or_max max" \
            --use_noise_ref false \
            --use_dereverb_ref false
        ./run.sh --stage 3 --stop_stage 4 --fs 16k --max_wav_duration 50 \
            --train_set "tr_mix_single_anechoic_max_16k" \
            --valid_set "cv_mix_single_anechoic_max_16k" \
            --test_sets "tt_mix_single_anechoic_max_16k" \
            --use_noise_ref false \
            --use_dereverb_ref false
        touch whamr.done
    fi

    cd "cwd"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Prepare additional data files for each dataset"

    # VoiceBank+DEMAND (1ch, 48kHz)
    rootdir="$(realpath ../../../egs2/vctk_noisy/enh1)"
    for x in tr_26spk cv_2spk tt_2spk; do
        mkdir -p data/vctk_noisy_${x}
        for f in utt2spk spk2utt; do
            ln -s "${rootdir}"/dump/raw/${x}/${f} data/vctk_noisy_${x}/
        done
        for f in wav.scp spk1.scp; do
            sed -e "s# dump/# ${rootdir}/dump/#g" "${rootdir}"/dump/raw/${x}/${f} > data/vctk_noisy_${x}/${f}
        done
        ln -s spk1.scp data/vctk_noisy_${x}/dereverb1.scp
        awk '{print $1 " 1ch_48k"}' data/vctk_noisy_${x}/utt2spk > data/vctk_noisy_${x}/utt2category
        awk '{print $1 " 48000"}' data/vctk_noisy_${x}/utt2spk > data/vctk_noisy_${x}/utt2fs
    done

    # DNS20-Challenge (1ch, 16kHz, noisy reverberant)
    rootdir="$(realpath ../../../egs2/dns_ins20/enh1)"
    for x in tr_synthetic cv_synthetic tt_synthetic_with_reverb tt_synthetic_no_reverb; do
        mkdir -p data/dns20_${x}
        for f in utt2spk spk2utt; do
            ln -s "${rootdir}"/dump/raw/${x}/${f} data/dns20_${x}/
        done
        for f in wav.scp spk1.scp; do
            sed -e "s# dump/# ${rootdir}/dump/#g" "${rootdir}"/dump/raw/${x}/${f} > data/dns20_${x}/${f}
        done
        ln -s spk1.scp data/dns20_${x}/dereverb1.scp
        awk '{print $1 " 1ch_16k"}' data/dns20_${x}/utt2spk > data/dns20_${x}/utt2category
        awk '{print $1 " 16000"}' data/dns20_${x}/utt2spk > data/dns20_${x}/utt2fs
    done

    # CHiME-4 (5ch, 16kHz, noisy)
    rootdir="$(realpath ../../../egs2/chime4/enh1)"
    for x in tr05_simu_isolated_6ch_track dt05_simu_isolated_6ch_track et05_simu_isolated_6ch_track; do
        mkdir -p data/chime4_${x}
        for f in utt2spk spk2utt; do
            ln -s "${rootdir}"/dump/raw/${x}/${f} data/chime4_${x}/
        done
        for f in wav.scp spk1.scp; do
            sed -e "s# dump/# ${rootdir}/dump/#g" "${rootdir}"/dump/raw/${x}/${f} > data/chime4_${x}/${f}
        done
        ln -s spk1.scp data/chime4_${x}/dereverb1.scp
        awk '{print $1 " 5ch_16k"}' data/chime4_${x}/utt2spk > data/chime4_${x}/utt2category
        awk '{print $1 " 16000"}' data/chime4_${x}/utt2spk > data/chime4_${x}/utt2fs
    done
    for x in dt05_real_isolated_6ch_track et05_real_isolated_6ch_track; do
        mkdir -p data/chime4_${x}
        for f in utt2spk spk2utt text; do
            ln -s "${rootdir}"/dump/raw/${x}/${f} data/chime4_${x}/
        done
        for f in wav.scp spk1.scp; do
            sed -e "s# dump/# ${rootdir}/dump/#g" "${rootdir}"/dump/raw/${x}/${f} > data/chime4_${x}/${f}
        done
        awk '{print $1 " 5ch_16k"}' data/chime4_${x}/utt2spk > data/chime4_${x}/utt2category
        awk '{print $1 " 16000"}' data/chime4_${x}/utt2spk > data/chime4_${x}/utt2fs
    done

    # REVERB (8ch, 16kHz, slighly noisy, reverberant)
    rootdir="$(realpath ../../../egs2/reverb/enh1)"
    for x in tr_simu_8ch_multich dt_simu_8ch_multich et_simu_8ch_multich; do
        mkdir -p data/reverb_${x}
        for f in utt2spk spk2utt; do
            ln -s "${rootdir}"/dump/raw/${x}/${f} data/reverb_${x}/
        done
        for f in wav.scp spk1.scp; do
            sed -e "s# dump/# ${rootdir}/dump/#g" "${rootdir}"/dump/raw/${x}/${f} > data/reverb_${x}/${f}
        done
        ln -s spk1.scp data/reverb_${x}/dereverb1.scp
        awk '{print $1 " 8ch_16k_reverb"}' data/reverb_${x}/utt2spk > data/reverb_${x}/utt2category
        awk '{print $1 " 16000"}' data/reverb_${x}/utt2spk > data/reverb_${x}/utt2fs
    done
    for x in dt_real_8ch_multich et_real_8ch_multich; do
        mkdir -p data/reverb_${x}
        for f in utt2spk spk2utt text; do
            ln -s "${rootdir}"/dump/raw/${x}/${f} data/reverb_${x}/
        done
        sed -e "s# dump/# ${rootdir}/dump/#g" "${rootdir}"/dump/raw/${x}/wav.scp > data/reverb_${x}/wav.scp
        # pseudo spk1.scp for convenience of SE evaluation
        ln -s wav.scp data/reverb_${x}/spk1.scp
        awk '{print $1 " 8ch_16k_reverb"}' data/reverb_${x}/utt2spk > data/reverb_${x}/utt2category
        awk '{print $1 " 16000"}' data/reverb_${x}/utt2spk > data/reverb_${x}/utt2fs
    done

    # WHAMR! (2ch, 16kHz, noisy, w/o reverberation)
    rootdir="$(realpath ../../../egs2/whamr/enh1)"
    for x in tr_mix_single_anechoic_max_16k cv_mix_single_anechoic_max_16k tt_mix_single_anechoic_max_16k; do
        mkdir -p data/whamr_${x}
        for f in utt2spk spk2utt; do
            ln -s "${rootdir}"/dump/raw/${x}/${f} data/whamr_${x}/
        done
        for f in wav.scp spk1.scp; do
            sed -e "s# dump/# ${rootdir}/dump/#g" "${rootdir}"/dump/raw/${x}/${f} > data/whamr_${x}/${f}
        done
        ln -s spk1.scp data/whamr_${x}/dereverb1.scp
        awk '{print $1 " 2ch_16k"}' data/whamr_${x}/utt2spk > data/whamr_${x}/utt2category
        awk '{print $1 " 16000"}' data/whamr_${x}/utt2spk > data/whamr_${x}/utt2fs
    done
    # WHAMR! (2ch, 16kHz, noisy, w/ reverberation)
    for x in tr_mix_single_reverb_max_16k cv_mix_single_reverb_max_16k tt_mix_single_reverb_max_16k; do
        mkdir -p data/whamr_${x}
        for f in utt2spk spk2utt; do
            ln -s "${rootdir}"/dump/raw/${x}/${f} data/whamr_${x}/
        done
        for f in wav.scp spk1.scp; do
            sed -e "s# dump/# ${rootdir}/dump/#g" "${rootdir}"/dump/raw/${x}/${f} > data/whamr_${x}/${f}
        done
        sed -e "s#/s\([0-9]\)_reverb/#/s\1_anechoic/#g" data/whamr_${x}/spk1.scp > data/whamr_${x}/dereverb1.scp
        awk '{print $1 " 2ch_16k_both"}' data/whamr_${x}/utt2spk > data/whamr_${x}/utt2category
        awk '{print $1 " 16000"}' data/whamr_${x}/utt2spk > data/whamr_${x}/utt2fs
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Combine all datasets"

    utils/combine_data.sh --skip_fix true --extra_files "utt2category utt2fs spk1.scp dereverb1.scp" \
        data/train_dns20_vctk_whamr_chime4_reverb \
        data/dns20_tr_synthetic \
        data/vctk_noisy_tr_26spk \
        data/whamr_tr_mix_single_anechoic_max_16k \
        data/whamr_tr_mix_single_reverb_max_16k \
        data/chime4_tr05_simu_isolated_6ch_track \
        data/reverb_tr_simu_8ch_multich

    # exclude REVERB from the valid set because it doesn't have strictly aligned references
    utils/combine_data.sh --skip_fix true --extra_files "utt2category utt2fs spk1.scp dereverb1.scp" \
        data/valid_dns20_vctk_whamr_chime4 \
        data/dns20_cv_synthetic \
        data/vctk_noisy_cv_2spk \
        data/whamr_cv_mix_single_anechoic_max_16k \
        data/whamr_cv_mix_single_reverb_max_16k \
        data/chime4_dt05_simu_isolated_6ch_track
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
