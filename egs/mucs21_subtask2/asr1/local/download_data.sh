#!/bin/bash -e

#Copyright



available_languages=(
    "hi" "mr" "or" "ta" "te" "gu" "hi-en" "bn-en"
)
db=$1
lang=$2

if [ $# != 2 ]; then
    echo "Usage: $0 <db_root_dir> <spk>"
    echo "Available langauges: ${available_languages[*]}"
    exit 1
fi

if ! $(echo ${available_languages[*]} | grep -q ${lang}); then
    echo "Specified langauge (${lang}) is not available or not supported." >&2
    echo "Choose from: ${available_languages[*]}"
    exit 1
fi

declare -A trainset
trainset['hi']='https://www.openslr.org/resources/103/Hindi_train.zip'
trainset['mr']='https://www.openslr.org/resources/103/Marathi_train.zip'
trainset['or']='https://www.openslr.org/resources/103/Odia_train.zip'
trainset['ta']='https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e'
trainset['te']='https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e'
trainset['gu']='https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e'
trainset['hi-en']='https://www.openslr.org/resources/104/Hindi-English_train.zip'
trainset['bn-en']='https://www.openslr.org/resources/104/Bengali-English_train.zip'

declare -A testset
testset['hi']='https://www.openslr.org/resources/103/Hindi_test.zip'
testset['mr']='https://www.openslr.org/resources/103/Marathi_test.zip'
testset['or']='https://www.openslr.org/resources/103/Odia_test.zip'
testset['ta']='https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e'
testset['te']='https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e'
testset['gu']='https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e'
testset['hi-en']='https://www.openslr.org/resources/104/Hindi-English_test.zip'
testset['bn-en']='https://www.openslr.org/resources/104/Bengali-English_test.zip'

cwd=`pwd`
if [ ! -e ${db}/${spk}.done ]; then
    mkdir -p ${db}
    cd ${db}
    mkdir -p ${lang}
    cd ${lang}
    wget ${testset[$lang]}
    # tar xf cmu_indic_${spk}.tar.bz2
    # rm cmu_indic_${spk}.tar.bz2
    cd $cwd
    echo "Successfully finished download."
    touch ${db}/${lang}.done
else
    echo "Already exists. Skip download."
fi
