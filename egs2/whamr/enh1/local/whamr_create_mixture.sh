#!/usr/bin/env bash

# Copyright  2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0

wham_noise=   # Path to the directory containing WHAM! noise
mono=False
min_or_max=min
sample_rate=8k

. utils/parse_options.sh
. path.sh
. cmd.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <dir> <wsj0-path> <wsj0-full-wav> <whamr-wav>"
  echo " where <dir> is download space,"
  echo " <wsj0-path> is the original wsj0 path"
  echo " <wsj0-full-wav> is wsj0 full wave files path, <whamr-wav> is wav generation space."
  echo "Note: this script won't actually re-download things if called twice,"
  echo "because we use the --continue flag to 'wget'."
  echo "Note: <wsj0-full-wav> contains all the wsj0 (or wsj) utterances in wav format,"
  echo "and the directory is organized according to"
  echo "  scripts/data/mix_2_spk_filenames_{tr,cv,tt}.csv"
  echo ", which are the mixture combination schemes."
  exit 1;
fi

dir=$1
wsj0_path=$2
wsj_full_wav=$3
whamr_wav=$4


wdir=data/local/downloads
mkdir -p ${dir}
mkdir -p ${wdir}
echo "Downloading WHAMR! data generation scripts and documentation."

if [ -z "$wham_noise" ]; then
  # 17.65 GB unzipping to 35 GB
  wham_noise_url=https://storage.googleapis.com/whisper-public/wham_noise.zip
  wget --continue -O $wdir/wham_noise.zip ${wham_noise_url}
  if [ $(ls ${dir}/wham_noise 2>/dev/null | wc -l) -eq 4 ]; then
    echo "'${dir}/wham_noise/' already exists. Skipping..."
  else
    unzip ${wdir}/wham_noise.zip -d ${dir}
  fi
  wham_noise=${dir}/wham_noise
fi

script_url=https://storage.googleapis.com/whisper-public/whamr_scripts.tar.gz
wget --continue -O $wdir/whamr_scripts.tar.gz ${script_url}
tar -xzf ${wdir}/whamr_scripts.tar.gz -C ${dir}

# If you want to generate both min and max versions with 8k and 16k data,
#  remove lines 59 and 60.
sed -i -e "s#MONO = True#MONO = ${mono}#" \
       -e "s#DATA_LEN = \['max', 'min'\]#DATA_LEN = ['${min_or_max}']#" \
       -e "s#SAMPLE_RATES = \['16k', '8k'\]#SAMPLE_RATES = ['${sample_rate}']#" \
       ${dir}/whamr_scripts/create_wham_from_scratch.py

echo "WSJ0 wav file."
local/convert2wav.sh ${wsj0_path} ${wsj_full_wav} || exit 1;

echo "Creating Mixtures."
if [ -z "$(python -m pip list | grep pyroomacoustics)" ]; then
  echo -e "Please install pyroomacoustics first:\n pip install pyroomacoustics==0.2.0"
  exit 1;
fi
# Run simulation (single-process)
# (This may take ~11 hours to generate min version, 8k data
#  on Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz)
cd ${dir}/whamr_scripts || exit 1
echo "Log is in ${dir}/whamr_scripts/mix.log"
${train_cmd} ${dir}/whamr_scripts/mix.log python create_wham_from_scratch.py \
  --wsj0-root ${wsj_full_wav} \
  --wham-noise-root ${wham_noise} \
  --output-dir ${whamr_wav}

# In the default configuration, the script will write about 444 GB of data:
#  - min_8k: 52 GB (mono=True) / 102 GB (mono=False)
#  - min_16k: ? GB
#  - max_8k: ? GB
#  - max_16k: ? GB
