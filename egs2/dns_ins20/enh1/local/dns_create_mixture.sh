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
total_hours=100
snr_lower=0
snr_upper=40

rm -r data/ 2>/dev/null || true
mkdir -p data/

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

  #outputs
  noisy_wav=${dns_wav}/noisy
  clean_wav=${dns_wav}/clean
  noise_wav=${dns_wav}/noise
  log_dir=data/log

  #modify the input paths for "\" separated paths
  sed -e "/^noisy_destination/s#.*#noisy_destination: ${noisy_wav}#g"  \
      -e "/^clean_destination/s#.*#clean_destination: ${clean_wav}#g"  \
      -e "/^noise_destination/s#.*#noise_destination: ${noise_wav}#g"  \
      -e "/^total_hours/s#.*#total_hours: ${total_hours}#g"  \
      -e "/^snr_lower/s#.*#snr_lower: ${snr_lower}#g"  \
      -e "/^snr_upper/s#.*#snr_upper: ${snr_upper}#g"  \
      -e "/^noise_dir/s#.*#noise_dir: ${noise_dir}#g"  \
      -e "/^speech_dir/s#.*#speech_dir: ${speech_dir}#g"  \
      -e "/^log_dir/s#.*#log_dir: ${log_dir}#g" ${configure} \
    > ${train_cfg}
else
  cp ${configure} ${train_cfg}
fi

mix_script=${dns}/noisyspeech_synthesizer_multiprocessing.py

if [ ! -f ${configure} -a -f ${mix_script} ]; then
  echo -e "Please check configurtion ${configure} and mix_script ${mix_script} exist"
  exit 1;
fi

# Default configuration will generate 33GB data under "${dns_wav}"
echo "Creating Mixtures for Training and Validation Data."
python ${mix_script} --cfg ${PWD}/${train_cfg} >/dev/null || exit 1;
