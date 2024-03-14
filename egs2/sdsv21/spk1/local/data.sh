#!/usr/bin/env bash

set -e
set -u
set -o pipefail

# For Voxceleb1 and 2 downloads, request the download links from https://mm.kaist.ac.kr/datasets/voxceleb/ and update the script

# Mozilla Commonvoice specfics
lang=fa # Farsi
cv_dir=commonvoice/cv-corpus-16.0-2023-12-06 # CV 16.0
data_url=''

stage=1
stop_stage=100
n_proc=8

data_dir_prefix= # root dir to save datasets.

trg_dir=data

. utils/parse_options.sh
. db.sh
. path.sh
. cmd.sh

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}



if [ -z ${data_dir_prefix} ]; then
    log "Root dir for dataset not defined, setting to ${MAIN_ROOT}/egs2/sdsv_2021"
    data_dir_prefix=${MAIN_ROOT}/egs2/sdsv_2021
else
    log "Root dir set to ${VOXCELEB}"
    data_dir_prefix=${VOXCELEB}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Text Data Download and Extract for VoxCelebs"
    if [ ! -x /usr/bin/wget ]; then
        log "Cannot execute wget. wget is required for download."
        exit 3
    fi

    # download Vox1-O eval protocol
    if [ ! -f "${data_dir_prefix}/veri_test2.txt" ]; then
        log "Download Vox1-O cleaned eval protocol."
        wget -P ${data_dir_prefix} https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
    else
       log "Skip downloading Vox1-O cleaned eval protocol."
    fi

    # download VoxCeleb1 devset txt data
    if [ ! -f ${data_dir_prefix}/vox1_dev_txt.zip ]; then
        wget -P  ${data_dir_prefix} https://mm.kaist.ac.kr/datasets/voxceleb/data/vox1_dev_txt.zip
    else
        log "Voxceleb1 devset txt data exists. Skip download."
    fi

    # download VoxCeleb1 testset txt data
    if [ ! -f ${data_dir_prefix}/vox1_test_txt.zip ]; then
        wget -P ${data_dir_prefix} https://mm.kaist.ac.kr/datasets/voxceleb/data/vox1_test_txt.zip
    else
        log "Voxceleb1 testset txt data exists. Skip download."
    fi

    # download VoxCeleb2 devset txt data
    if [ ! -f ${data_dir_prefix}/vox2_dev_txt.zip ]; then
        # -c for the case when download is incomplete
        # (to continue download when the script is ran again)
        wget -P ${data_dir_prefix} -c https://mm.kaist.ac.kr/datasets/voxceleb/data/vox2_dev_txt.zip
    else
        log "Voxceleb2 devset txt data exists. Skip download."
    fi


    if [ -d ${data_dir_prefix}/txt ]; then
        rm -rf ${data_dir_prefix}/txt
    fi

    log "Extracting VoxCeleb1 test set text data."
    if [ ! -d "${data_dir_prefix}/voxceleb1/test" ]; then
        unzip -q ${data_dir_prefix}/vox1_test_txt.zip -d ${data_dir_prefix}
        mkdir -p ${data_dir_prefix}/voxceleb1/test
        mv ${data_dir_prefix}/txt ${data_dir_prefix}/voxceleb1/test
    else
        log "voxceleb1/test exists. Skip extraction."
    fi

    log "Extracting VoxCeleb1 development set text data."
    if [ ! -d ${data_dir_prefix}/voxceleb1/dev ]; then
        unzip -q ${data_dir_prefix}/vox1_dev_txt.zip -d ${data_dir_prefix}
        mkdir -p ${data_dir_prefix}/voxceleb1/dev
        mv ${data_dir_prefix}/txt ${data_dir_prefix}/voxceleb1/dev
    else
        log "voxceleb1/dev. Skip extraction."
    fi

    log "Extracting VoxCeleb2 text data."
    if [ ! -d ${data_dir_prefix}/voxceleb2/dev ]; then
        unzip -q ${data_dir_prefix}/vox2_dev_txt.zip -d ${data_dir_prefix}
        mkdir -p ${data_dir_prefix}/voxceleb2/dev
        mv ${data_dir_prefix}/txt ${data_dir_prefix}/voxceleb2/dev
    else
        log "voxceleb2/dev. Skip extraction."
   fi

    log "Stage 1, DONE."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Convert Voxcelebs text data to audio by crawling YouTube."
    if [ ! -d "${data_dir_prefix}/voxceleb1/test" ]; then
        python local/crawl_voxcelebs_mp.py --root_dir ${data_dir_prefix}/voxceleb1/test --n_proc ${n_proc}
    else
        log "Skipping VoxCeleb1 test set crawling because 'voxceleb1' folder exists. If you want to crawl voxceleb1 (e.g., stopped during crawling), remove 'voxceleb1/test' folder"
    fi

    if [ ! -d ${data_dir_prefix}/voxceleb1/dev ]; then
        python local/crawl_voxcelebs_mp.py --root_dir ${data_dir_prefix}/voxceleb1/dev --n_proc ${n_proc}
    else
        log "Skipping VoxCeleb1 development set crawling because 'voxceleb1' folder exists. If you want to crawl voxceleb1 (e.g., stopped during crawling), remove 'voxceleb1/dev' folder"
    fi


    if [ ! -d ${data_dir_prefix}/voxceleb2/dev ]; then
        python local/crawl_voxcelebs_mp.py --root_dir ${data_dir_prefix}/voxceleb2/dev --n_proc ${n_proc}
    else
        log "Skipping VoxCeleb2 crawling because 'voxceleb2/dev' folder exists. If you want to crawl voxceleb2 (e.g., stopped during crawling), remove 'voxceleb2/dev' folder"
    fi
    log "Stage 2, DONE."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Download LibriSpeech train-other-500 data"

    if [ ! -f ${data_dir_prefix}/train-other-500.tar.gz ]; then
        wget -P ${data_dir_prefix} -c https://www.openslr.org/resources/12/train-other-500.tar.gz
    else
        log "train-other-500.tar.gz exists. Skip download."
    fi

    if [ -d ${data_dir_prefix}/LibriSpeech ]; then
        log "LibriSpeech directory exists. Skip extracting LibriSpeech"
    else
        log "Extracting LibriSpeech data."
        tar -zxvf ${data_dir_prefix}/train-other-500.tar.gz -C ${data_dir_prefix}
    fi

    # Changing from flac to wav
    if [ -d "${data_dir_prefix}/librispeech_wav" ]; then
        log "librispeech_wav exists. Skip converting librispeech"
    else
        log "Converting librispeech to wav"
        mkdir -p ${data_dir_prefix}/librispeech_wav
        python local/librispeech_flac2wav.py --src "${data_dir_prefix}/LibriSpeech/train-other-500" --dst "${data_dir_prefix}/librispeech_wav" --n_proc ${n_proc}
    fi
    log "Stage 3, DONE."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Download Mozilla Common Voice version"
    log "First obtain a personal download link from https://commonvoice.mozilla.org/en/datasets"

    # Downloading the commonvoice corpus
    if [ -d ${data_dir_prefix}/commonvoice ]; then
        log "Commonvoice exists. Skip downloading Commonvoice"
    else
        mkdir -p ${data_dir_prefix}/commonvoice
        wget -P ${data_dir_prefix}/commonvoice -c ${data_url}
        for file in "${data_dir_prefix}/commonvoice/"*tar.gz*; do
            mv "$file" "${file%%\?*}"
        done
    fi
    # Extracting the commonvoice corpus
    if [ -d "${data_dir_prefix}/${cv_dir}/${lang}" ]; then
        log "Skip extracting commonvoice"
    else
        log "Extracting commonvoice"
        for file in "${data_dir_prefix}/commonvoice/"*tar.gz*; do
            tar -xzvf "$file" -C ${data_dir_prefix}/commonvoice/
        done
    fi
    # Changing from mp3 to wav and changing to Kaldi style features
    if [ -d "${trg_dir}" ]; then
        log "Skip converting commonvoice"
    else
        log "Converting commonvoice and making Kaldi-style feature."
        mkdir -p ${trg_dir}
        for part in "validated" "test" "dev"; do
            python local/commonvoice_data_prep.py "${data_dir_prefix}/${cv_dir}/${lang}" ${part} ${trg_dir}/"$(echo "cv_${part}_${lang}" | tr - _)"
        done
    fi
    log "Stage 4, DONE."
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Download Musan and RIR_NOISES for augmentation."

    if [ ! -f ${data_dir_prefix}/rirs_noises.zip ]; then
        wget -P ${data_dir_prefix} -c http://www.openslr.org/resources/28/rirs_noises.zip
    else
        log "RIRS_NOISES exists. Skip download."
    fi

    if [ ! -f ${data_dir_prefix}/musan.tar.gz ]; then
        wget -P ${data_dir_prefix} -c http://www.openslr.org/resources/17/musan.tar.gz
    else
        log "Musan exists. Skip download."
    fi

    if [ -d ${data_dir_prefix}/RIRS_NOISES ]; then
        log "Skip extracting RIRS_NOISES"
    else
        log "Extracting RIR augmentation data."
        unzip -q ${data_dir_prefix}/rirs_noises.zip -d ${data_dir_prefix}
    fi

    if [ -d ${data_dir_prefix}/musan ]; then
        log "Skip extracting Musan"
    else
        log "Extracting Musan noise augmentation data."
        tar -zxvf ${data_dir_prefix}/musan.tar.gz -C ${data_dir_prefix}
    fi

    # make scp files
    log "Making scp files for musan"
    for x in music noise speech; do
        find ${data_dir_prefix}/musan/${x} -iname "*.wav" > ${trg_dir}/musan_${x}.scp
    done

    # Use small and medium rooms, leaving out largerooms.
    # Similar setup to Kaldi and VoxCeleb_trainer.
    log "Making scp files for RIRS_NOISES"
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/mediumroom -iname "*.wav" > ${trg_dir}/rirs.scp
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/smallroom -iname "*.wav" >> ${trg_dir}/rirs.scp
    log "Stage 5, DONE."
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Change into kaldi-style feature."
    if [ ! -d "${trg_dir}/combined_train_set" ]; then
        mkdir -p ${trg_dir}/voxceleb1_test
        mkdir -p ${trg_dir}/voxceleb1_dev
        mkdir -p ${trg_dir}/voxceleb2_dev
        mkdir -p ${trg_dir}/librispeech_train_other_500
        python local/data_prep.py --src "${data_dir_prefix}/voxceleb1/test" --dst "${trg_dir}/voxceleb1_test"
        python local/data_prep.py --src "${data_dir_prefix}/voxceleb1/dev" --dst "${trg_dir}/voxceleb1_dev"
        python local/data_prep.py --src "${data_dir_prefix}/voxceleb2/dev" --dst "${trg_dir}/voxceleb2_dev"
        python local/data_prep.py --src "${data_dir_prefix}/librispeech_wav" --dst "${trg_dir}/librispeech_train_other_500"

        for f in wav.scp utt2spk spk2utt; do
            sort ${trg_dir}/voxceleb1_test/${f} -o ${trg_dir}/voxceleb1_test/${f}
            sort ${trg_dir}/voxceleb1_dev/${f} -o ${trg_dir}/voxceleb1_dev/${f}
            sort ${trg_dir}/voxceleb2_dev/${f} -o ${trg_dir}/voxceleb2_dev/${f}
            sort ${trg_dir}/librispeech_train_other_500/${f} -o ${trg_dir}/librispeech_train_other_500/${f}
        done

        # combine VoxCeleb 1 & 2 dev sets, librispeech train_other_500, and Commonvoice 16 Farsi for combined training set.
        mkdir -p ${trg_dir}/combined_train_set
        for f in wav.scp utt2spk spk2utt; do
            {
                cat "${trg_dir}/voxceleb1_dev/${f}"
                cat "${trg_dir}/voxceleb2_dev/${f}"
                cat "${trg_dir}/librispeech_train_other_500/${f}"
                cat "${trg_dir}/cv_validated_fa/${f}"
            } >> "${trg_dir}/combined_train_set/${f}.tmp"

            sort "${trg_dir}/combined_train_set/${f}.tmp" -o "${trg_dir}/combined_train_set/${f}"
            rm -f "${trg_dir}/combined_train_set/${f}.tmp" # cleanup tmp file
        done


        # make test trial compatible with ESPnet.
        python local/convert_trial.py --trial ${data_dir_prefix}/veri_test2.txt --scp ${trg_dir}/voxceleb1_test/wav.scp --out ${trg_dir}/voxceleb1_test

    else
        log "Skipping stage 6 because "${trg_dir}/combined_train_set" exists"
    fi
    log "Stage 6, DONE."

fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Download and prepare DeepMine sample data"
    if [ ! -f "${data_dir_prefix}/SampleDeepMine.tar.gz" ]; then
        wget -P ${data_dir_prefix} https://data.deepmine.ir/en/files/SampleDeepMine.tar.gz
    else
        log "Skipping the SampleDeepmine tar download because "${data_dir_prefix}/SampleDeepMine.tar.gz" exists"
    fi

    if [ ! -d "${data_dir_prefix}/SampleDeepMine" ]; then
        tar -zxvf ${data_dir_prefix}/SampleDeepMine.tar.gz -C ${data_dir_prefix}
    else
        log "Skipping untar SampleDeepMine because "${data_dir_prefix}/SampleDeepMine" exists"
    fi

    if [ ! -d "${trg_dir}/SampleDeepMine" ]; then
        mkdir -p ${trg_dir}/SampleDeepMine
        log "Making SampleDeepMine wav.scp utt2spk and spk2utt files"
        python local/sampledeepmine_prep.py --src "${data_dir_prefix}/SampleDeepMine/wav/" --dst "${trg_dir}/SampleDeepMine" --input_file "${data_dir_prefix}/SampleDeepMine/wav/files.txt"
        for f in wav.scp utt2spk spk2utt; do
            sort ${trg_dir}/SampleDeepMine/${f} -o ${trg_dir}/SampleDeepMine/${f}
        done

        # Generate trial file
        log "Generating SampleDeepMine trial txt file"
        python local/generate_trial_file.py --src "${data_dir_prefix}/SampleDeepMine/wav/files.txt" --dst "${data_dir_prefix}/veri_test_SampleDeepmine.txt"
        # make test trial compatible with ESPnet
        log "Making the SampleDeepMine trial compatible with ESPnet"
        python local/convert_trial.py --trial ${data_dir_prefix}/veri_test_SampleDeepmine.txt --scp ${trg_dir}/SampleDeepMine/wav.scp --out ${trg_dir}/SampleDeepMine

    else
        log "Skipping preparing SampleDeepmine for trial because "${trg_dir}/SampleDeepMine" exists"
    fi

    log "Stage 7, DONE."
fi
