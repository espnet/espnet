#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
nj=32
outdir=aishell4_simu

. utils/parse_options.sh || exit 1;
outdir=$(realpath "$outdir")

help_message=$(cat << EOF
Usage: $0 [--stage <stage>] [--stop_stage <stop_stage>] [--nj <nj>] [--outdir <outdir>]

  optional argument:
    [--stage]: 0 (default) to 5
    [--stop_stage]: 0 to 100 (default)
    [--nj]: number of parallel processes for data simulation
    [--outdir]: output directory for storing simulated data (default is "aishell4_simu")

  expected directory structure
        <AISHELL4>
         |-- test/
         |   |-- TextGrid/
         |   |   |-- L_*.rttm
         |   |   |-- M_*.rttm
         |   |   |-- S_*.rttm
         |   |   |-- L_*.TextGrid
         |   |   |-- M_*.TextGrid
         |   |   \-- S_*.TextGrid
         |   \-- wav/
         |       |-- L_*.flac
         |       |-- M_*.flac
         |       \-- S_*.flac
         |
         |-- train_L/
         |   |-- TextGrid/
         |   |   |-- *_L_*.rttm
         |   |   \-- *_L_*.TextGrid
         |   \-- wav/
         |       \-- *_L_*.flac
         |
         |-- train_M/
         |   |-- TextGrid/
         |   |   |-- *_M_*.rttm
         |   |   \-- *_M_*.TextGrid
         |   \-- wav/
         |       \-- *_M_*.flac
         |
         \-- train_S/
             |-- TextGrid/
             |   |-- *_S_*.rttm
             |   \-- *_S_*.TextGrid
             \-- wav/
                 \-- *_S_*.flac
EOF
)

if [ $# -gt 0 ]; then
    log "${help_message}"
    exit 2
fi


if [ -z "${AISHELL4}" ]; then
    log "Fill the value of 'AISHELL4' in db.sh"
    log "(available at https://www.openslr.org/111/)"
    exit 1
fi

if [ ! -e "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' in db.sh"
    log "(available at http://openslr.org/12/)"
    exit 1
elif [ ! -e "${LIBRISPEECH}/train-clean-100" ]; then
    log "Please ensure '${LIBRISPEECH}/train-clean-100' exists"
    exit 1
elif [ ! -e "${LIBRISPEECH}/train-clean-360" ]; then
    log "Please ensure '${LIBRISPEECH}/train-clean-360' exists"
    exit 1
elif [ ! -e "${LIBRISPEECH}/dev-clean" ]; then
    log "Please ensure '${LIBRISPEECH}/dev-clean' exists"
    exit 1
fi

if [ ! -e "${MUSAN}" ]; then
    log "Fill the value of 'MUSAN' in db.sh"
    log "(available at http://openslr.org/17/)"
    exit 1
elif [ ! -e "${MUSAN}/noise" ]; then
    log "Please ensure '${MUSAN}/noise' exists"
    exit 1
elif [ ! -e "${MUSAN}/music" ]; then
    log "Please ensure '${MUSAN}/music' exists"
    exit 1
fi

if [ ! -e "${AUDIOSET}" ]; then
    log "Fill the value of 'AUDIOSET' in db.sh"
    log "(available at https://github.com/marc-moreaux/audioset_raw)"
    exit 1
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Prepare simulation enviroment"

    # Prepare pyrirgen
    if [ ! -d "local/pyrirgen" ]; then
        git clone https://github.com/phecda-xu/RIR-Generator.git local/pyrirgen
        # based on commit ab038671b238fdd8d71df8dabe64137d86947b57
        patch -i local/pyrirgen.pyx.patch local/pyrirgen/pyrirgen.pyx
        log "pyrirgen successfully downloaded"
    fi

    python -m pip install -r local/pyrirgen/requirements.txt
    curdir=$PWD
    cd local/pyrirgen
    make
    cd "$curdir"
    log "pyrirgen successfully compiled"

    log "Downloading AISHELL4 repository"
    URL=https://github.com/felixfuyihui/AISHELL-4.git

    if [ ! -d "${outdir}/AISHELL-4" ] ; then
        git clone "$URL" "${outdir}/AISHELL-4"
        # based on commit bad82b77c3753df1b232c5c6491cd3e2f2e32d24
        patch -i local/generate_rir_trainingdata.py.patch "${outdir}"/AISHELL-4/data_preparation/generate_rir_trainingdata.py
        patch -i local/generate_isotropic_noise.py.patch "${outdir}"/AISHELL-4/data_preparation/generate_isotropic_noise.py
        patch -i local/generate_fe_trainingdata.py.patch "${outdir}"/AISHELL-4/data_preparation/generate_fe_trainingdata.py
        log "git successfully downloaded"
    fi

    python -m pip install -r "${outdir}/AISHELL-4"/requirements.txt

    # overwrite pyrirgen-related libraries provided in the AISHELL-4 repository,
    # because they were bound to a specific Python version
    rm "${outdir}"/AISHELL-4/data_preparation/librirgen.so
    rm "${outdir}"/AISHELL-4/data_preparation/pyrirgen.*.so
    ln -s "${curdir}"/local/pyrirgen/librirgen.so "${outdir}"/AISHELL-4/data_preparation/
    ln -s "${curdir}"/local/pyrirgen/pyrirgen.*.so "${outdir}"/AISHELL-4/data_preparation/
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] ; then  
    log "Stage 1: Simulate RIRs"

    mkdir -p "${outdir}/data/rirs"
    mkdir -p "${outdir}/data/log"
    export LD_LIBRARY_PATH="${outdir}/AISHELL-4/data_preparation":$LD_LIBRARY_PATH

    curdir=$PWD
    # You may want to manually modify this file if you made modifications to conf/slurm.conf
    sed -e 's/--export=PATH$/--export=PATH,LD_LIBRARY_PATH/' "${curdir}/../../TEMPLATE/enh1/conf/slurm.conf" > "${outdir}/data/log/slurm.conf"
    cd "${outdir}"/AISHELL-4/data_preparation/ || exit 1
    # This takes ~6.5 hours with Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHZ (nj=32).
    # In total, ~4.5 GB data will be generated.
    ${train_cmd} --config "${outdir}/data/log/slurm.conf" JOB=1:${nj} "${outdir}"/data/log/rir.JOB.log \
        python ./generate_rir_trainingdata.py JOB ${nj} \
            --output_dir "${outdir}/data/rirs" \
            --seed 1
    cd "$curdir"

    # This should generate 12500 RIRs for 2500 rooms.
    # Each RIR data has shape (24, 8000). The 0-8, 8-16, and 16-24 channels correspond to three sources, respectively.
    find "${outdir}/data/rirs" -iname "*.wav" | sort > "${outdir}/data/rirs.lst"
    num_rir=$(<"${outdir}/data/rirs.lst" wc -l)
    if [ ${num_rir} -ne 12500 ]; then
        log "Error: Expected 12500 wav files, but got ${num_rir}"
        exit 1
    fi

    # 12500 = 11875 (train) + 625 (dev)
    # This script will generate train_rirs.lst and dev_rirs.lst under "${outdir}/data/"
    python local/split_train_dev_by_prefix.py "${outdir}/data/rirs.lst" \
        --num_dev "1/20" \
        --outfile "${outdir}/data/{}_rirs.lst" \
        --delim "_" \
        --prefix_num 3 \
        --mode "same_size_group"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Simulate isotropic noise"

    mkdir -p "${outdir}/data/isotropic_noise"
    # This takes ~1 hour with Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz.
    # 200 wav files (~2.9 GB) will be generated.
    # Each noise sample has shape (8, 960000).
    python "${outdir}"/AISHELL-4/data_preparation/generate_isotropic_noise.py \
        --output_dir "${outdir}/data/isotropic_noise" \
        --wavnum 200 \
        --seed 1

    # sort in numerically increasing order
    find "${outdir}/data/isotropic_noise" -iname "*.wav" | \
        awk -F"${outdir}/data/isotropic_noise/isotropic_" '{print $0, $2}' | \
        sort -k2 -n | cut -d' ' -f1 > "${outdir}/data/isotropic_noise.lst"

    head -n 190 "${outdir}/data/isotropic_noise.lst" > "${outdir}/data/train_isotropic_noise.lst"
    tail -n 10 "${outdir}/data/isotropic_noise.lst" > "${outdir}/data/dev_isotropic_noise.lst"
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Prepare speech and noise lists"

    ####################################################
    # prepare speech list for training and development #
    ####################################################
    find "${LIBRISPEECH}/train-clean-100/" "${LIBRISPEECH}/train-clean-360/" -iname "*.flac" | sort > "${outdir}/data/train_speech.lst"
    find "${LIBRISPEECH}/dev-clean/" -iname "*.flac" | sort > "${outdir}/data/dev_speech.lst"

    # This script will generate train_train_speech.lst and dev_train_speech.lst under "${outdir}/data/"
    python local/split_train_dev_by_prefix.py "${outdir}/data/train_speech.lst" \
        --num_dev "1/2" \
        --outfile "${outdir}/data/{}_train_speech.lst" \
        --delim "-" \
        --prefix_num 2 \
        --mode "similar_size_group"

    # This script will generate train_dev_speech.lst and dev_dev_speech.lst under "${outdir}/data/"
    python local/split_train_dev_by_prefix.py "${outdir}/data/dev_speech.lst" \
        --num_dev "1/2" \
        --outfile "${outdir}/data/{}_dev_speech.lst" \
        --delim "-" \
        --prefix_num 2 \
        --mode "similar_size_group"

    mv "${outdir}/data/train_train_speech.lst" "${outdir}/data/train_speech_spk1.lst"
    mv "${outdir}/data/dev_train_speech.lst" "${outdir}/data/train_speech_spk2.lst"
    mv "${outdir}/data/train_dev_speech.lst" "${outdir}/data/dev_speech_spk1.lst"
    mv "${outdir}/data/dev_dev_speech.lst" "${outdir}/data/dev_speech_spk2.lst"

    ####################################################
    # prepare noise lists for training and development #
    ####################################################
    wget -O "${outdir}/data/audioset.name" https://raw.githubusercontent.com/ConferencingSpeech/ConferencingSpeech2021/master/selected_lists/train/audioset.name
    wget -O "${outdir}/data/musan.name" https://raw.githubusercontent.com/ConferencingSpeech/ConferencingSpeech2021/master/selected_lists/train/musan.name

    # append category information to the raw audio list
    sed -i -e 's/\(\(\w\+\-\)\+\w\+\)-[0-9]\+\.wav/\0 \1/g' "${outdir}/data/musan.name"
    python local/prepare_audioset_category_list.py \
        "${outdir}/data/audioset.name" \
        --audioset_dir "$AUDIOSET" \
        --output_file "${outdir}/data/audioset.name"

    python local/prepare_data_list.py \
        --outfile "${outdir}/data/musan.lst" \
        --audiodirs "$MUSAN" \
        --audio-format "wav" \
        "${outdir}/data/musan.name"

    python local/prepare_data_list.py \
        --outfile "${outdir}/data/audioset.lst" \
        --audiodirs "$AUDIOSET" \
        --audio-format "wav" \
        --ignore-missing-files True \
        "${outdir}/data/audioset.name"

    # This script will generate train_musan.lst and dev_musan.lst under "${outdir}/data/"
    # 988 = 945 (train) + 43 (dev)
    python local/split_train_dev_by_column.py "${outdir}/data/musan.lst" \
        --num_dev "43" \
        --outfile "${outdir}/data/{}_musan.lst" \
        --mode "similar_size_group"

    # This script will generate train_audioset.lst and dev_audioset.lst under "${outdir}/data/"
    # 22418 = 21297 (train) + 1121 (dev)
    python local/split_train_dev_by_column.py "${outdir}/data/audioset.lst" \
        --num_dev "1/20" \
        --outfile "${outdir}/data/{}_audioset.lst" \
        --mode "similar_size_group"

    cat "${outdir}"/data/train_{musan,audioset}.lst > "${outdir}/data/train_noise.lst"
    cat "${outdir}"/data/dev_{musan,audioset}.lst > "${outdir}/data/dev_noise.lst"
    rm "${outdir}/data"/{train,dev}_musan.lst "${outdir}/data"/{train,dev}_audioset.lst
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Simulate training and development data"

    log "Simulating training data"
    mkdir -p "${outdir}/data/wavs/train"
    # This takes ~20.5 hours with Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz.
    # 60000 * 3 (mix, spk1, spk2) wav files (~128 GB) will be generated.
    # Each audio sample (16 kHz) has 8 channels.
    python "${outdir}"/AISHELL-4/data_preparation/generate_fe_trainingdata.py \
        --spk1_list "${outdir}/data/train_speech_spk1.lst" \
        --spk2_list "${outdir}/data/train_speech_spk2.lst" \
        --noise_list "${outdir}/data/train_noise.lst" \
        --rir_list "${outdir}/data/train_rirs.lst" \
        --isotropic_list "${outdir}/data/train_isotropic_noise.lst" \
        --mode "train" \
        --output_dir "${outdir}/data/wavs" \
        --wavnum 60000

    log "Simulating development data"
    mkdir -p "${outdir}/data/wavs/dev"
    # This takes ~1 hour with Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz.
    # 3000 * 3 (mix, spk1, spk2) wav files (~6.2 GB) will be generated.
    # Each audio sample (16 kHz) has 8 channels.
    python "${outdir}"/AISHELL-4/data_preparation/generate_fe_trainingdata.py \
        --spk1_list "${outdir}/data/dev_speech_spk1.lst" \
        --spk2_list "${outdir}/data/dev_speech_spk2.lst" \
        --noise_list "${outdir}/data/dev_noise.lst" \
        --rir_list "${outdir}/data/dev_rirs.lst" \
        --isotropic_list "${outdir}/data/dev_isotropic_noise.lst" \
        --mode "dev" \
        --output_dir "${outdir}/data/wavs" \
        --wavnum 3000
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Prepare training and development data"

    mkdir -p data/train
    find "${outdir}/data/wavs/train/mix" -iname "*.wav" > "${outdir}/data/wavs/train/mix.lst"
    sed -e 's/\.\(wav\|flac\)//' "${outdir}/data/wavs/train/mix.lst" | \
        awk -F '/' '{print $NF}' > "${outdir}/data/wavs/train/mix_id.lst"
    paste -d' ' "${outdir}/data/wavs/train/mix_id.lst" "${outdir}/data/wavs/train/mix.lst" | sort -u > data/train/wav.scp
    sed -e "s#${outdir}/data/wavs/train/mix/#${outdir}/data/wavs/train/spk1/#g" data/train/wav.scp > data/train/spk1.scp
    sed -e "s#${outdir}/data/wavs/train/mix/#${outdir}/data/wavs/train/spk2/#g" data/train/wav.scp > data/train/spk2.scp
    awk '{print $1, "dummy"}' data/train/wav.scp > data/train/utt2spk
    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
    utils/validate_data_dir.sh --no-feats --no-text data/train
    rm "${outdir}/data/wavs/train/mix_id.lst"

    mkdir -p data/dev
    find "${outdir}/data/wavs/dev/mix" -iname "*.wav" > "${outdir}/data/wavs/dev/mix.lst"
    sed -e 's/\.\(wav\|flac\)//' "${outdir}/data/wavs/dev/mix.lst" | \
        awk -F '/' '{print $NF}' > "${outdir}/data/wavs/dev/mix_id.lst"
    paste -d' ' "${outdir}/data/wavs/dev/mix_id.lst" "${outdir}/data/wavs/dev/mix.lst" | sort -u > data/dev/wav.scp
    sed -e "s#${outdir}/data/wavs/dev/mix/#${outdir}/data/wavs/dev/spk1/#g" data/dev/wav.scp > data/dev/spk1.scp
    sed -e "s#${outdir}/data/wavs/dev/mix/#${outdir}/data/wavs/dev/spk2/#g" data/dev/wav.scp > data/dev/spk2.scp
    awk '{print $1, "dummy"}' data/dev/wav.scp > data/dev/utt2spk
    utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
    utils/validate_data_dir.sh --no-feats --no-text data/dev
    rm "${outdir}/data/wavs/dev/mix_id.lst"
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Prepare the evaluation data for the Speaker Independent task"

    find "${AISHELL4}/test/wav" -iname "*.flac" > "${outdir}"/data/test_wav.lst
    find "${AISHELL4}/test/TextGrid" -iname "*.TextGrid" > "${outdir}"/data/test_TextGrid.lst

    mkdir -p "${outdir}/data/wavs/test"
    # This takes ~12 minutes with Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz.
    # 10381 wav files (~11 GB) will be generated.
    # Each audio sample (16 kHz) has 8 channels.
    python "${outdir}"/AISHELL-4/data_preparation/generate_nospk_testdata.py \
        --wav_list "${outdir}/data/test_wav.lst" \
        --textgrid_list "${outdir}/data/test_TextGrid.lst" \
        --output_dir "${outdir}/data/wavs"

    mkdir -p data/test
    find "${outdir}/data/wavs/test" -iname "*.wav" > "${outdir}/data/wavs/test/wav.lst"
    sed -e 's/\.\(wav\|flac\)//' "${outdir}/data/wavs/test/wav.lst" | \
        awk -F '/' '{print $NF}' > "${outdir}/data/wavs/test/wav_id.lst"
    paste -d' ' "${outdir}/data/wavs/test/wav_id.lst" "${outdir}/data/wavs/test/wav.lst" | sort -u > data/test/wav.scp
    ln -s wav.scp data/test/spk1.scp
    ln -s wav.scp data/test/spk2.scp
    awk '{print $1, "dummy"}' data/test/wav.scp > data/test/utt2spk
    utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
    utils/validate_data_dir.sh --no-feats --no-text data/test
    rm "${outdir}/data/wavs/test/wav_id.lst"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
