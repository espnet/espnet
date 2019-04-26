#!/bin/bash

# Copyright  2018  Johns Hopkins University (Author: Xuankai Chang)

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <dir> <wsj0-path> <wsj0-full-wav> <wsj0-2mix-wav>"
  echo " where <dir> is download space,"
  echo " <wsj0-path> is the original wsj0 path"
  echo " <wsj0-full-wav> is wsj0 full wave files path, <wsj0-2mix-wav> is wav generation space."
  echo "Note: this script won't actually re-download things if called twice,"
  echo "because we use the --continue flag to 'wget'."
  echo "Note: this script can be used to create wsj0_2mix and wsj_2mix corpus"
  echo "Note: <wsj0-full-wav> contains all the wsj0 (or wsj) utterances in wav format,"
  echo "and the directory is organized according to"
  echo "  scripts/mix_2_spk_tr.txt, scripts/mix_2_spk_cv.txt and mix_2_spk_tt.txt"
  echo ", which are the mixture combination schemes."
  exit 1;
fi

dir=$1
wsj0_path=$2
wsj_full_wav=$3
wsj_2mix_wav=$4

rootdir=$PWD
echo "Downloading WSJ0_mixture scripts."

url=http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip
wdir=data/local/downloads

mkdir -p ${dir}
mkdir -p ${wdir}/log

wget --continue -O $wdir/create-speaker-mixtures.zip ${url}

unzip ${wdir}/create-speaker-mixtures.zip -d ${dir}

sed -i -e "s=/db/processed/public/WSJ0WAV_full=${wsj_full_wav}=" \
       -e "s=/mm1/leroux/wsj0-mix/2speakers=${wsj_2mix_wav}=" \
       -e "s='min','max'='max'=" \
       ${dir}/create_wav_2speakers.m

echo "WSJ0 wav file."
local/convert2wav.sh ${wsj0_path} ${wsj_full_wav} || exit 1;

echo "Creating Mixtures."

matlab_cmd="matlab -nojvm -nodesktop -nodisplay -nosplash -r create_wav_2speakers"

mixfile=${dir}/mix_matlab.sh
echo "#!/bin/bash" > $mixfile
echo $matlab_cmd >> $mixfile
chmod +x $mixfile

# Run Matlab
cd ${dir}
$train_cmd ${dir}/mix.log $mixfile

cd ${rootdir}
