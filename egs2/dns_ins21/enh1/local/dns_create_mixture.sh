#!/usr/bin/env bash

. utils/parse_options.sh
. path.sh
. cmd.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <dns> <dns_wav> "
  echo " where <dns> is dns directory,"
  echo " <dns_wav> is wav generation space."
  exit 1;
fi

dns=$1
dns_wav=$2

rm -r data/ 2>/dev/null || true
mkdir -p data/


# modify path in the original noisyspeech_synthesizer.cfg
configure=${dns}/noisyspeech_synthesizer.cfg
train_cfg=data/noisyspeech_synthesizer.cfg


if [ ! -f ${configure} ]; then
  echo -e "Please check configurtion ${configure} exist"
  exit 1;
fi

# input datas
noise_dir=${dns}/datasets/noise
speech_dir=${dns}/datasets/clean/read_speech

# additional clean datas
clean_singing=${dns}/datasets/clean/singing_voice
clean_emotion=${dns}/datasets/clean/emotional_speech
clean_mandarin=${dns}/datasets/clean/mandarin_speech

# acoustic params
rir_table_csv=${dns}/datasets/acoustic_params_wideband/RIR_table_simple.csv
clean_speech_t60_csv=${dns}/datasets/acoustic_params_wideband/cleanspeech_table_t60_c50.csv

# outputs
noisy_wav=${dns_wav}/noisy
clean_wav=${dns_wav}/clean
noise_wav=${dns_wav}/noise
log_dir=log

# modify the input paths for "\" separated paths
sed -e "/^noisy_destination/s#.*#noisy_destination:${noisy_wav}#g"  \
    -e "/^clean_destination/s#.*#clean_destination:${clean_wav}#g"  \
    -e "/^noise_destination/s#.*#noise_destination:${noise_wav}#g"  \
    -e "/^noise_dir/s#.*#noise_dir:${noise_dir}#g"  \
    -e "/^speech_dir/s#.*#speech_dir:${speech_dir}#g"  \
    -e "/^clean_singing/s#.*#clean_singing:${clean_singing}#g"  \
    -e "/^clean_emotion/s#.*#clean_emotion:${clean_emotion}#g"  \
    -e "/^clean_mandarin/s#.*#clean_mandarin:${clean_mandarin}#g"  \
    -e "/^rir_table_csv/s#.*#rir_table_csv:${rir_table_csv}#g"  \
    -e "/^clean_speech_t60_csv/s#.*#clean_speech_t60_csv:${clean_speech_t60_csv}#g"  \
    -e "/^log_dir/s#.*#log_dir:${log_dir}#g" ${configure} \
  > ${train_cfg}

# modify the path separator
sed -i -e 's:\\:/:g' ${rir_table_csv}

mix_script=${dns}/noisyspeech_synthesizer_singleprocess.py

if [ ! -f ${configure} -a -f ${mix_script} ]; then
  echo -e "Please check configurtion ${configure} and mix_script ${mix_script} exist"
  exit 1;
fi

echo "Creating Mixtures for Training and Validation Data."
python ${mix_script} --cfg ${PWD}/${train_cfg} >/dev/null || exit 1;
