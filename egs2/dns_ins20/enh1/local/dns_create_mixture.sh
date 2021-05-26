#!/usr/bin/env bash

configure=   # Path to the configure file

. utils/parse_options.sh
. path.sh
. cmd.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 --configure <configure> <dns> <dns_wav> "
  echo " where <dns> is dns directory,"
  echo " <dns_wav> is wav generation space."
  exit 1;
fi

dns=$1
dns_wav=$2

rm -r data/ 2>/dev/null || true
mkdir -p data/

mod_conf=true
if [ -z "$configure" ]; then
  # modify path in the original noisyspeech_synthesizer.cfg
  configure=${dns}/noisyspeech_synthesizer.cfg
  train_cfg=data/noisyspeech_synthesizer.cfg


  if [ ! -f ${configure} ]; then
    echo -e "Please check configurtion ${configure} exist"
    exit 1;
  fi

  #input datas
  noise_dir=${dns}/datasets/noise
  speech_dir=${dns}/datasets/clean
  noise_dir=/data/DNS-challenge/DNS//datasets/noise
  speech_dir=/data/DNS-challenge/DNS//datasets/clean

  #outputs
  noisy_wav=${dns_wav}/noisy
  clean_wav=${dns_wav}/clean
  noise_wav=${dns_wav}/noise
  log_dir=data/log

  #modify the input paths for "\" separated paths
  sed -e "/^noisy_destination/s#.*#noisy_destination:${noisy_wav}#g" ${configure} | \
  sed -e "/^clean_destination/s#.*#clean_destination:${clean_wav}#g" | \
  sed -e "/^noise_destination/s#.*#noise_destination:${noise_wav}#g" | \
  sed -e "/^noise_dir/s#.*#noise_dir:${noise_dir}#g" | \
  sed -e "/^speech_dir/s#.*#speech_dir:${speech_dir}#g" | \
  sed -e "/^log_dir/s#.*#log_dir:${log_dir}#g" \
    > ${train_cfg}

fi

mix_script=${dns}/noisyspeech_synthesizer_multiprocessing.py

if [ ! -f ${configure} -a -f ${mix_script} ]; then
  echo -e "Please check configurtion ${configure} and mix_script ${mix_script} exist"
  exit 1;
fi

echo "Creating Mixtures for Training and Validation Data."
python ${mix_script} --cfg $PWD/${train_cfg} >/dev/null || exit 1;










# ####################
# wdir=data/local/downloads
# mkdir -p ${dir}
# mkdir -p ${wdir}
# echo "Downloading WHAMR! data generation scripts and documentation."

# if [ -z "$wham_noise" ]; then
#   # 17.65 GB unzipping to 35 GB
#   wham_noise_url=https://storage.googleapis.com/whisper-public/wham_noise.zip
#   wget --continue -O $wdir/wham_noise.zip ${wham_noise_url}
#   if [ $(ls ${dir}/wham_noise 2>/dev/null | wc -l) -eq 4 ]; then
#     echo "'${dir}/wham_noise/' already exists. Skipping..."
#   else
#     unzip ${wdir}/wham_noise.zip -d ${dir}
#   fi
#   wham_noise=${dir}/wham_noise
# fi

# script_url=https://storage.googleapis.com/whisper-public/whamr_scripts.tar.gz
# wget --continue -O $wdir/whamr_scripts.tar.gz ${script_url}
# tar -xzf ${wdir}/whamr_scripts.tar.gz -C ${dir}

# # If you want to generate both min and max versions with 8k and 16k data,
# #  remove lines 59 and 60.
# sed -i -e "s#MONO = True#MONO = ${mono}#" \
#        -e "s#DATA_LEN = \['max', 'min'\]#DATA_LEN = ['${min_or_max}']#" \
#        -e "s#SAMPLE_RATES = \['16k', '8k'\]#SAMPLE_RATES = ['${sample_rate}']#" \
#        ${dir}/whamr_scripts/create_wham_from_scratch.py

# echo "WSJ0 wav file."
# local/convert2wav.sh ${wsj0_path} ${wsj_full_wav} || exit 1;

# echo "Creating Mixtures."
# if [ -z "$(python -m pip list | grep pyroomacoustics)" ]; then
#   echo -e "Please install pyroomacoustics first:\n pip install pyroomacoustics==0.2.0"
#   exit 1;
# fi
# # Run simulation (single-process)
# # (This may take ~11 hours to generate min version, 8k data
# #  on Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz)
# cd ${dir}/whamr_scripts || exit 1
# echo "Log is in ${dir}/whamr_scripts/mix.log"
# ${train_cmd} ${dir}/whamr_scripts/mix.log python create_wham_from_scratch.py \
#   --wsj0-root ${wsj_full_wav} \
#   --wham-noise-root ${wham_noise} \
#   --output-dir ${whamr_wav}

# # In the default configuration, the script will write about 444 GB of data:
# #  - min_8k: 52 GB (mono=True) / 102 GB (mono=False)
# #  - min_16k: ? GB
# #  - max_8k: ? GB
# #  - max_16k: ? GB
