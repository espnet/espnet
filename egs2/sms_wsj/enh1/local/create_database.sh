#!/usr/bin/env bash

# Copyright  2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0
set -e
set -u
set -o pipefail

nj=16
min_or_max=max
sample_rate=8k
num_spk=2
download_rir=true
write_all=true

. utils/parse_options.sh
. ./path.sh

if [[ "$min_or_max" != "max" ]]; then
  echo "Error: min_or_max must be max: ${min_or_max}"
  exit 1
fi
if [[ "$sample_rate" == "16k" ]]; then
  sample_rate2=16000
  echo "Warning: sample_rate=16k is not officially supported yet."
  exit 1
elif [[ "$sample_rate" == "8k" ]]; then
  sample_rate2=8000
else
  echo "Error: sample rate must be either 16k or 8k: ${sample_rate}"
  exit 1
fi
if [[ $num_spk != [2-4] ]]; then
  echo "Error: number of speakers must be 2, 3, or 4: ${num_spk}"
  exit 1
fi
if [[ "$download_rir" != "true" ]] && [[ "$download_rir" != "false" ]]; then
  echo "Error: download_rir must be either true or false: ${download_rir}"
  exit 1
fi
if [[ "$write_all" != "true" ]] && [[ "$write_all" != "false" ]]; then
  echo "Error: write_all must be either true or false: ${write_all}"
  exit 1
fi

if [ $# -ne 4 ]; then
  echo "Usage: $0 <wsj0-path> <wsj1-path> <wsj-zeromean-wav> <sms-wsj-wav>"
  echo "  where <wsj0-path> is the original wsj0 path,"
  echo "  <wsj1-path> is the original wsj1 path,"
  echo "  <wsj-zeromean-wav> is path to store the zero-mean normalized wsj."
  echo "  <sms-wsj-wav> is path to store the generated sms-wsj."
  echo "[Optional]"
  echo "  --nj <nj>                     # number of parallel jobs (Default=${nj})"
  echo "  --min-or-max <min_or_max>     # min or max length for generating mixtures (Default=${min_or_max})"
  echo "  --sample-rate <sample_rate>   # sample rate (Default=${sample_rate})"
  echo "  --num-spk <num_spk>           # number of speakers (Default=${num_spk})"
  echo "  --download-rir <download_rir> # whether to download or simulate RIRs (Default=${download_rir})"
  echo "  --write-all <download_rir>    # whether to store all intermediate audio data (Default=${write_all})"
  echo "Note: this script won't actually re-download things if called twice,"
  echo "because we use the --continue flag to 'wget'."
  exit 1;
fi

wsj0_path=$1
wsj1_path=$2
wsj_zeromean_wav=$3
sms_wsj_wav=$4
json_dir=${sms_wsj_wav}
rir_dir=${sms_wsj_wav}/rirs


sph2pipe=${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe
if ! command -v "${sph2pipe}" &> /dev/null; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

if ! command -v "mpiexec" &> /dev/null; then
  echo "Could not find (or execute) the mpiexec program";
  exit 1;
fi

if ! command -v "sox" &> /dev/null; then
  echo "Could not find (or execute) the sox program";
  exit 1;
fi


# This takes about 15 minutes with nj=16.
if [[ ! -d ${wsj_zeromean_wav} ]]; then
  echo "Creating zero-mean normalized wsj at '${wsj_zeromean_wav}'."
  mpiexec -np ${nj} python -m sms_wsj.database.wsj.write_wav \
      with dst_dir=${wsj_zeromean_wav} wsj0_root=${wsj0_path} \
      wsj1_root=${wsj1_path} sample_rate=${sample_rate2}
fi


mkdir -p ${json_dir}
if [[ ! -f ${json_dir}/wsj_${sample_rate}_zeromean.json ]]; then
  echo "Creating ${json_dir}/wsj_${sample_rate}_zeromean.json"
  python -m sms_wsj.database.wsj.create_json \
    with json_path=${json_dir}/wsj_${sample_rate}_zeromean.json \
    database_dir=${wsj_zeromean_wav} as_wav=True
fi


if [[ ! -d ${rir_dir} ]]; then
  if ${download_rir}; then
    mkdir -p ${rir_dir}
    echo "Downloading RIRs (50.8 GB) in '${rir_dir}'"
    # wget -qO- https://zenodo.org/record/3517889/files/sms_wsj.tar.gz.parta{a,b,c,d,e} \
    #   | tar -C ${rir_dir}/ -zx --checkpoint=10000 --checkpoint-action=echo="%u/5530000 %c"

    ## In case of instable network connection, please use the following command:
    temp_dir=$(mktemp -d data/temp.XXX) || exit 1
    for url in https://zenodo.org/record/3517889/files/sms_wsj.tar.gz.parta{a,b,c,d,e}; do
      wget --continue -O "${temp_dir}/$(basename ${url})" ${url}
    done
    cat ${temp_dir}/sms_wsj.tar.gz.parta* | \
      tar -C ${rir_dir}/ -zx --checkpoint=10000 --checkpoint-action=echo="%u/5530000 %c"
    rm -rf "${temp_dir}"
  else
    echo "Simulating RIRs in '${rir_dir}'"
    # This takes around 1900 / (ncpus - 1) hours.
    mpiexec -np ${nj} python -m sms_wsj.database.create_rirs \
      with database_path=${rir_dir}
  fi
fi


if [[ ! -f ${json_dir}/intermediate_sms_wsj.json ]]; then
  echo "Creating ${json_dir}/intermediate_sms_wsj.json"
  python -m sms_wsj.database.create_intermediate_json \
    with json_path=${json_dir}/intermediate_sms_wsj.json rir_dir=${rir_dir} \
    wsj_json_path=${json_dir}/wsj_${sample_rate}_zeromean.json debug=False num_speakers=${num_spk}
fi


# This takes about 25 minutes with the default configuration.
# NOTE (Wangyou): If you try to rerun this part, please make sure the directories under
#   ${sms_wsj_wav}/ are deleted in advance.
echo "Creating ${sms_wsj_wav} audio data in '${sms_wsj_wav}'"
mpiexec -np ${nj} python -m sms_wsj.database.write_files \
  with dst_dir=${sms_wsj_wav} json_path=${json_dir}/intermediate_sms_wsj.json \
  write_all=True debug=False


if [[ ! -f ${json_dir}/sms_wsj.json ]]; then
  echo "Creating ${json_dir}/sms_wsj.json"
  python -m sms_wsj.database.create_json_for_written_files \
    with db_dir=${sms_wsj_wav} intermed_json_path=${json_dir}/intermediate_sms_wsj.json \
    write_all=True json_path=${json_dir}/sms_wsj.json debug=False
fi


# The total disk usage of SMS-WSJ is 442.1 GiB + 240.2 GiB = 682.3 GiB.
# This number may be larger than the officially reported one, because we write
#  all intermediate files (see [additional data] below) to the disk.
# --------------------------------------------------------------------------------
# directory/file  disk usage  #channels   #samples
# --------------------------------------------------------------------------------
# tail            120.1 GiB   6           35875 * 2 (only when write_all=True)
# early           120.1 GiB   6           35875 * 2 (only when write_all=True)
# observation     60.0 GiB    6           35875
# noise           60.0 GiB    6           35875
# --------------------------- [additional data] ----------------------------------
# source_signal   120.1 GiB   6           35875 * 2
# reverb_source   120.1 GiB   6           35875 * 2
# --------------------------------------------------------------------------------
# rirs            52.6 GiB    6           143500=(33561+982+1332)*4 (up to 4 srcs)
# wsj_8k_zeromean 29.2 GiB    1           131824
# --------------------------------------------------------------------------------
