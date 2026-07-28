#!/usr/bin/env bash

configure=   # Path to the configure file

. utils/parse_options.sh
. path.sh
. cmd.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 --configure <configure> <ms_snsd> <ms_snsd_wav> "
  echo " where <ms_snsd> is ms_snsd directory,"
  echo " <ms_snsd_wav> is wav generation space."
  exit 1;
fi

ms_snsd=$1
ms_snsd_wav=$2

# You can tune these later
total_hours_train=30 # increase if needed
total_hours_test=1  # increase if needed
snr_lower=0
snr_upper=40
total_snrlevels=5
audio_length=10
silence_length=0.2
sampling_rate=16000
audioformat="*.wav"

rm -r data/ 2>/dev/null || true
mkdir -p data/

mix_script=${ms_snsd}/noisyspeech_synthesizer.py
base_cfg=${configure:-${ms_snsd}/noisyspeech_synthesizer.cfg}

# Inputs (match your repo layout)
speech_dir_train=${ms_snsd}/clean_train
noise_dir_train=${ms_snsd}/noise_train
speech_dir_test=${ms_snsd}/clean_test
noise_dir_test=${ms_snsd}/noise_test

for p in "${mix_script}" "${base_cfg}"; do
  [ -f "${p}" ] || { echo "Error: not found: ${p}"; exit 1; }
done
for d in "${speech_dir_train}" "${noise_dir_train}" "${speech_dir_test}" "${noise_dir_test}"; do
  [ -d "${d}" ] || { echo "Error: not found dir: ${d}"; exit 1; }
done

mkdir -p "${ms_snsd_wav}"

make_cfg () {
  local dst_cfg=$1
  local speech_dir=$2
  local noise_dir=$3
  local total_hours=$4

  # MS-SNSD cfg only supports these keys (no output destination keys!)
  cat > "${dst_cfg}" << EOF
[noisy_speech]
sampling_rate: ${sampling_rate}
audioformat: ${audioformat}
audio_length: ${audio_length}
silence_length: ${silence_length}
total_hours: ${total_hours}
snr_lower: ${snr_lower}
snr_upper: ${snr_upper}
total_snrlevels: ${total_snrlevels}
noise_dir: ${noise_dir}
speech_dir: ${speech_dir}
noise_types_excluded: None
EOF
}

run_one () {
  local tag=$1              # train | test
  local speech_dir=$2
  local noise_dir=$3
  local total_hours=$4
  local out_root=$5         # ${ms_snsd_wav}/train or /test

  local workdir=${ms_snsd_wav}/work_${tag}
  rm -rf "${workdir}"
  mkdir -p "${workdir}"

  # Copy required python files into workdir so outputs are created under workdir
  cp "local/ms_snsd_noisyspeech_synthesizer.py" "${workdir}/noisyspeech_synthesizer.py"
  # MS-SNSD script imports audiolib.py (and possibly others). Copy at least audiolib.py.
  cp "${ms_snsd}/audiolib.py" "${workdir}/"

  local cfg=${workdir}/cfg.cfg
  make_cfg "${cfg}" "${speech_dir}" "${noise_dir}" "${total_hours}"

  (cd "${workdir}" && python noisyspeech_synthesizer.py --cfg cfg.cfg --cfg_str noisy_speech)

  # The script writes fixed folder names under workdir:
  #   NoisySpeech_training, CleanSpeech_training, Noise_training
  mkdir -p "${out_root}"
  rm -rf "${out_root}/noisy" "${out_root}/clean" "${out_root}/noise" 2>/dev/null || true
  mv "${workdir}/NoisySpeech_training" "${out_root}/noisy"
  mv "${workdir}/CleanSpeech_training" "${out_root}/clean"
  mv "${workdir}/Noise_training"       "${out_root}/noise"
}

echo "[Stage1] Generate TRAIN mixtures"
run_one train "${speech_dir_train}" "${noise_dir_train}" "${total_hours_train}" "${ms_snsd_wav}/train"

echo "[Stage1] Generate TEST mixtures"
run_one test  "${speech_dir_test}"  "${noise_dir_test}"  "${total_hours_test}"  "${ms_snsd_wav}/test"
