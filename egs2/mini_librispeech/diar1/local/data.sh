#!/bin/bash

# Copyright 2020 Johns Hopkins University (Jiatong Shi)
# Adopted from https://github.com/hitachi-speech/EEND
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

stage=1
stop_stage=1
SECONDS=0

# simulation options
simu_opts_overlap=yes
simu_opts_num_speaker=2
simu_opts_sil_scale=2
simu_opts_rvb_prob=0.5
simu_opts_num_train=500

# simulation source
mini_librispeech_url=http://www.openslr.org/resources/31
rir_url=http://www.openslr.org/resources/26/sim_rir_8k.zip
noise_url=https://www.openslr.org/resources/17/

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z "${MINI_LIBRISPEECH}" ]; then
    log "Fill the value of 'MINI_LIBRISPEECH' of db.sh"
    exit 1
fi
mkdir -p "${MINI_LIBRISPEECH}"

simu_dirs=(
"${MINI_LIBRISPEECH}"/diarization-data
)

. ./utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo " Stage 1: prepare simulation sources"
    local/download_and_untar.sh "${MINI_LIBRISPEECH}" "${mini_librispeech_url}"  dev-clean-2
    local/download_and_untar.sh "${MINI_LIBRISPEECH}" "${mini_librispeech_url}" train-clean-5
    if [ ! -f "${MINI_LIBRISPEECH}"/dev_clean_2.done ]; then
        local/data_prep.sh "${MINI_LIBRISPEECH}"/LibriSpeech/dev-clean-2 data/dev_clean_2 || exit 1
        touch "${MINI_LIBRISPEECH}"/dev_clean_2.done
    fi
    if [ ! -f "${MINI_LIBRISPEECH}"/train_clean_5.done ]; then
        local/data_prep.sh "${MINI_LIBRISPEECH}"/LibriSpeech/train-clean-5 data/train_clean_5 || exit 1
        touch "${MINI_LIBRISPEECH}"/train_clean_5.done
    fi
    if [ ! -f "${MINI_LIBRISPEECH}"/noise.done ]; then
        mkdir -p data/noise
        local/download_and_untar.sh "${MINI_LIBRISPEECH}" "${noise_url}" musan
        find "${MINI_LIBRISPEECH}"/musan/noise/free-sound -iname "*.wav" \
            | awk '{n=split($1,A,/[\/\.]/); print A[n-3]"_"A[n-1], $1}' \
            | sort > data/noise/wav.scp
        awk '{print $1, $1}' data/noise/wav.scp > data/noise/utt2spk
        utils/fix_data_dir.sh data/noise
        touch "${MINI_LIBRISPEECH}"/noise.done
    fi
    if [ ! -f "${MINI_LIBRISPEECH}"/simu_rirs_8k.done ]; then
        mkdir -p data/simu_rirs_8k
        if [ ! -e "${MINI_LIBRISPEECH}"/sim_rir_8k.zip ]; then
            wget -nv --no-check-certificate "${rir_url}" -P "${MINI_LIBRISPEECH}"
        fi
        unzip -q "${MINI_LIBRISPEECH}"/sim_rir_8k.zip -d "${MINI_LIBRISPEECH}"/sim_rir_8k
        find "${MINI_LIBRISPEECH}"/sim_rir_8k -iname "*.wav" \
            | awk '{n=split($1,A,/[\/\.]/); print A[n-3]"_"A[n-1], $1}' \
            | sort > data/simu_rirs_8k/wav.scp
        awk '{print $1, $1}' data/simu_rirs_8k/wav.scp > data/simu_rirs_8k/utt2spk
        utils/fix_data_dir.sh data/simu_rirs_8k
        touch "${MINI_LIBRISPEECH}"/simu_rirs_8k.done
    fi
fi

simudir=data/simu
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "simulation of mixture"
    mkdir -p "${simudir}"/.work
    random_mixture_cmd=random_mixture_nooverlap.py
    make_mixture_cmd=make_mixture_nooverlap.py
    if [ "$simu_opts_overlap" == "yes" ]; then
        random_mixture_cmd=random_mixture.py
        make_mixture_cmd=make_mixture.py
    fi

    for dset in train_clean_5 dev_clean_2; do
        n_mixtures="$simu_opts_num_train"
        simuid="${dset}"_ns"${simu_opts_num_speaker}"_beta"${simu_opts_sil_scale}"_"${n_mixtures}"
        # check if you have the simulation
        if ! validate_data_dir.sh --no-text --no-feats "${simudir}"/data/"${simuid}"; then
            echo "generate random mixture"
            # random mixture generation
            ${train_cmd} "${simudir}"/.work/random_mixture_"${simuid}".log \
                "$random_mixture_cmd" --n_speakers "$simu_opts_num_speaker" --n_mixtures "$n_mixtures" \
                --speech_rvb_probability "$simu_opts_rvb_prob" \
                --sil_scale "$simu_opts_sil_scale" \
                data/"$dset" data/noise data/simu_rirs_8k \
                \> "$simudir"/.work/mixture_"$simuid".scp
            nj=10
            mkdir -p "$simudir"/wav/"$simuid"
            # distribute simulated data to $simu_dirs
            split_scps=
            for n in $(seq $nj); do
                split_scps="$split_scps ${simudir}/.work/mixture_${simuid}.$n.scp"
                mkdir -p "${simudir}"/.work/data_"${simuid}".$n
                actual=${simu_dirs[($n-1)%${#simu_dirs[@]}]}/$simuid.$n
                mkdir -p $actual

                ln -nfs "$(realpath $actual)" $simudir/wav/"${simuid}"/$n
            done
            utils/split_scp.pl "${simudir}"/.work/mixture_"${simuid}".scp $split_scps || exit 1

            ${train_cmd} JOB=1:$nj "${simudir}"/.work/make_mixture_"${simuid}".JOB.log \
                $make_mixture_cmd --rate=8000 \
                "${simudir}"/.work/mixture_"${simuid}".JOB.scp \
                "${simudir}"/.work/data_"${simuid}".JOB "${simudir}"/wav/"${simuid}"/JOB \
                || { cat "${simudir}"/.work/make_mixture_"${simuid}".*.log; exit 1; }
            utils/combine_data.sh "${simudir}"/data/"${simuid}" "${simudir}"/.work/data_"${simuid}".*
            steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                "${simudir}"/data/"${simuid}"/utt2spk "${simudir}"/data/"${simuid}"/segments \
                "${simudir}"/data/"${simuid}"/rttm
            utils/data/get_reco2dur.sh "${simudir}"/data/"${simuid}"
        fi
    done
fi
