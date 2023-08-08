#!/usr/bin/env bash
set -e
set -u
set -o pipefail

stage=1
stop_stage=100
n_proc=8

data_dir_prefix=/home/jeeweonj/corpora/voxcelebs # root dir to save datasets.

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
    log "Root dir for dataset not defined, setting to ${MAIN_ROOT}/egs2/voxceleb"
    data_dir_prefix=${MAIN_ROOT}/egs2/voxceleb
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Text Data Download and Extract"
    if [ ! -x /usr/bin/wget ]; then
        echo "Cannot execute wget. wget is required for download."
        exit 3
    fi

    # download Vox1-O eval protocol
    if [ ! -f "${data_dir_prefix}/veri_test2.txt" ]; then
        echo "Download Vox1-O cleaned eval protocol."
        wget -P ${data_dir_prefix} https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
    else
       echo "Skip downloading Vox1-O cleaned eval protocol."
    fi

    # download VoxCeleb1 devset txt data
    if [ ! -f ${data_dir_prefix}/vox1_dev_txt.zip ]; then
        wget -P ${data_dir_prefix} https://mm.kaist.ac.kr/datasets/voxceleb/data/vox1_dev_txt.zip
    else
        echo "Voxceleb1 devset txt data exists. Skip download."
    fi

    # download VoxCeleb1 testset txt data
    if [ ! -f ${data_dir_prefix}/vox1_test_txt.zip ]; then
        wget -P ${data_dir_prefix} https://mm.kaist.ac.kr/datasets/voxceleb/data/vox1_test_txt.zip
    else
        echo "Voxceleb1 testset txt data exists. Skip download."
    fi

    # download VoxCeleb2 devset txt data
    if [ ! -f ${data_dir_prefix}/vox2_dev_txt.zip ]; then
        # -c for the case when download is incomplete
        # (to continue download when the script is ran again)
        wget -P ${data_dir_prefix} -c https://mm.kaist.ac.kr/datasets/voxceleb/data/vox2_dev_txt.zip
    else
        echo "Voxceleb2 devset txt data exists. Skip download."
    fi


    if [ -d ${data_dir_prefix}/txt ]; then
        rm -rf ${data_dir_prefix}/txt
    fi

    echo "Extracting VoxCeleb1 test set text data."
    unzip -q ${data_dir_prefix}/vox1_test_txt.zip -d ${data_dir_prefix}
    if [ ! -d "${data_dir_prefix}/voxceleb1/test" ]; then
        mkdir -p ${data_dir_prefix}/voxceelb1/test
    fi
    mv ${data_dir_prefix}/txt ${data_dir_prefix}/voxceleb1/test

    echo "Extracting VoxCeleb1 development set text data."
    unzip -q ${data_dir_prefix}/vox1_dev_txt.zip -d ${data_dir_prefix}
    if [ ! -d ${data_dir_prefix}/voxceleb1/dev ]; then
        mkdir -p ${data_dir_prefix}/voxceelb1/dev
    fi
    mv ${data_dir_prefix}/txt ${data_dir_prefix}/voxceleb1/dev

    echo "Extracting VoxCeleb2 text data."
    unzip -q ${data_dir_prefix}/vox2_dev_txt.zip -d ${data_dir_prefix}
    if [ ! -d ${data_dir_prefix}/voxceleb2/dev ]; then
        mkdir -p ${data_dir_prefix}/voxceelb2/dev
   fi
    mv ${data_dir_prefix}/txt ${data_dir_prefix}/voxceleb2/dev

    echo "Stage 1, DONE."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Convert text data to audio by crawling YouTube."
    if [ ! -d "${data_dir_prefix}/voxceleb1/test" ]; then
        python local/crawl_voxcelebs_mp.py --root_dir ${data_dir_prefix}/voxceleb1/test --n_proc ${n_proc}
    else
        echo "Skipping VoxCeleb1 test set crawling because 'voxceleb1' folder exists. If you want to crawl voxceleb1 (e.g., stopped during crawling), remove 'voxceleb1/test' folder"
    fi

    if [ ! -d ${data_dir_prefix}/voxceleb1/dev ]; then
        python local/crawl_voxcelebs_mp.py --root_dir ${data_dir_prefix}/voxceleb1/dev --n_proc ${n_proc}
    else
        echo "Skipping VoxCeleb1 development set crawling because 'voxceleb1' folder exists. If you want to crawl voxceleb1 (e.g., stopped during crawling), remove 'voxceleb1/dev' folder"
    fi


    if [ ! -d ${data_dir_prefix}/voxceleb2/dev ]; then
        python local/crawl_voxcelebs_mp.py --root_dir ${data_dir_prefix}/voxceleb2/dev --n_proc ${n_proc}
    else
        echo "Skipping VoxCeleb2 crawling because 'voxceleb2/dev' folder exists. If you want to crawl voxceleb2 (e.g., stopped during crawling), remove 'voxceleb2/dev' folder"
    fi
    echo "Stage 2, DONE."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Download Musan and RIR_NOISES for augmentation."

    if [ ! -f ${data_dir_prefix}/rirs_noises.zip ]; then
        wget -P ${data_dir_prefix} -c http://www.openslr.org/resources/28/rirs_noises.zip
    else
        echo "RIRS_NOISES exists. Skip download."
    fi

    if [ ! -f ${data_dir_prefix}/musan.tar.gz ]; then
        wget -P ${data_dir_prefix} -c http://www.openslr.org/resources/17/musan.tar.gz
    else
        echo "Musan exists. Skip download."
    fi

    if [ -d ${data_dir_prefix}/RIRS_NOISES ]; then
        echo "Skip extracting RIRS_NOISES"
    else
        echo "Extracting RIR augmentation data."
        unzip -q ${data_dir_prefix}/rirs_noises.zip -d ${data_dir_prefix}
    fi

    if [ -d ${data_dir_prefix}/musan ]; then
        echo "Skip extracting Musan"
    else
        echo "Extracting Musan noise augmentation data."
        tar -zxvf ${data_dir_prefix}/musan.tar.gz -C ${data_dir_prefix}
    fi

    # make scp files
    for x in music noise speech; do
        find ${data_dir_prefix}/musan/${x} -iname "*.wav" > ${trg_dir}/musan_${x}.scp
    done

    # Use small and medium rooms, leaving out largerooms.
    # Similar setup to Kaldi and VoxCeleb_trainer.
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/mediumroom -iname "*.wav" > ${trg_dir}/rirs.scp
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/smallroom -iname "*.wav" >> ${trg_dir}/rirs.scp
    echo "Stage 3, DONE."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Change into kaldi-style feature."
    mkdir -p ${trg_dir}/voxceleb1_test
    mkdir -p ${trg_dir}/voxceleb1_dev
    mkdir -p ${trg_dir}/voxceleb2_dev
    python local/data_prep.py --src "${data_dir_prefix}/voxceleb1/test" --dst "${trg_dir}/voxceleb1_test"
    python local/data_prep.py --src "${data_dir_prefix}/voxceleb1/dev" --dst "${trg_dir}/voxceleb1_dev"
    python local/data_prep.py --src "${data_dir_prefix}/voxceleb2/dev" --dst "${trg_dir}/voxceleb2_dev"

    for f in wav.scp utt2spk spk2utt; do
        sort ${trg_dir}/voxceleb1_test/${f} -o ${trg_dir}/voxceleb1_test/${f}
        sort ${trg_dir}/voxceleb1_dev/${f} -o ${trg_dir}/voxceleb1_dev/${f}
        sort ${trg_dir}/voxceleb2_dev/${f} -o ${trg_dir}/voxceleb2_dev/${f}
    done

    # combine VoxCeleb 1 and 2 dev sets for combined training set.
    mkdir -p ${trg_dir}/voxceleb12_devs
    for f in wav.scp utt2spk spk2utt; do
        cat ${trg_dir}/voxceleb1_dev/${f} >> ${trg_dir}/voxceleb12_devs/${f}
        cat ${trg_dir}/voxceleb2_dev/${f} >> ${trg_dir}/voxceleb12_devs/${f}
        sort ${trg_dir}/voxceleb12_devs/${f} -o ${trg_dir}/voxceleb12_devs/${f}
    done

    # make test trial compatible with ESPnet.
    python local/convert_trial.py --trial ${data_dir_prefix}/veri_test2.txt --scp ${trg_dir}/voxceleb1_test/wav.scp --out ${trg_dir}/voxceleb1_test

    echo "Stage 4, DONE."

fi
