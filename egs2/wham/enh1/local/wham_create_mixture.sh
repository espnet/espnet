#!/usr/bin/env bash

# Copyright  2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0

wsj0_2mix=    # Path to an existing wsj0-2mix data root directory
wham_noise=   # Path to the directory containing WHAM! noise
min_or_max=min
sample_rate=8k

. utils/parse_options.sh
. path.sh
. cmd.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <dir> <wsj0-path> <wsj0-full-wav> <wham-wav>"
  echo " where <dir> is download space,"
  echo " <wsj0-path> is the original wsj0 path"
  echo " <wsj0-full-wav> is wsj0 full wave files path, <wham-wav> is wav generation space."
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
wham_wav=$4


rootdir=$PWD
wdir=data/local/downloads
mkdir -p ${dir}
mkdir -p ${wdir}/log
echo "Downloading WHAM! data generation scripts and documentation."
if [ ! -d "$wham_noise" ]; then
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

if [ ! -d "${dir}/wham_scripts" ]; then
  script_url=https://storage.googleapis.com/whisper-public/wham_scripts.tar.gz
  wget --continue -O $wdir/wham_scripts.tar.gz ${script_url}
  tar -xzf ${wdir}/wham_scripts.tar.gz -C ${dir}
fi

# If you want to generate both min and max versions with 8k and 16k data,
#  comment out the following 3 lines.
sed -i -e "s#for datalen_dir in \['max', 'min'\]:#for datalen_dir in ['${min_or_max}']:#" \
       -e "s#for sr_dir in \['16k', '8k'\]:#for sr_dir in ['${sample_rate}']:#" \
       ${dir}/wham_scripts/create_wham_from_scratch.py

echo "WSJ0 wav file."
local/convert2wav.sh ${wsj0_path} ${wsj_full_wav} || exit 1;

echo "Creating Mixtures."
cd ${dir}/wham_scripts || exit 1
echo "Log is in ${dir}/wham_scripts/mix.log"

# Run simulation (single-process)
if [ -d "$wsj0_2mix" ]; then
  # (This may take ~6 hours to generate min version, 8k data
  #  on Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz)
  echo "Using existing wsj0-2mix data in $wsj0_2mix"
  ${train_cmd} ${dir}/wham_scripts/mix.log python create_wham_from_wsjmix.py \
    --wsjmix-dir-16k ${wsj0_2mix}/wav16k \
    --wsjmix-dir-8k ${wsj0_2mix}/wav8k \
    --wham-noise-root ${wham_noise} \
    --output-dir ${wham_wav}
else
  # (This may take ~3 hours to generate min version, 8k data
  #  on Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz)
  ${train_cmd} ${dir}/wham_scripts/mix.log python create_wham_from_scratch.py \
    --wsj0-root ${wsj_full_wav} \
    --wham-noise-root ${wham_noise} \
    --output-dir ${wham_wav}
fi

# In the default configuration, the script will write about 243 GB of data:
#  - min_8k: 28 GB
#  - min_16k: 56 GB
#  - max_8k: 53 GB
#  - max_16k: 106 GB
