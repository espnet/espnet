#!/bin/bash

# Copyright  2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0

nj=16
min_or_max=min
sample_rate=8k
download_rir=true

. utils/parse_options.sh
. ./cmd.sh
. ./path.sh

if [[ "$min_or_max" != "max" ]] && [[ "$min_or_max" != "min" ]]; then
  echo "Error: min_or_max must be either max or min: ${min_or_max}"
  exit 1
fi
if [[ "$sample_rate" == "16k" ]]; then
  sample_rate=16000
elif [[ "$sample_rate" == "8k" ]]; then
  sample_rate=8000
else
  echo "Error: sample rate must be either 16k or 8k: ${sample_rate}"
  exit 1
fi
if [[ "$download_rir" != "true" ]] && [[ "$download_rir" != "false" ]]; then
  echo "Error: download_rir must be either true or false: ${download_rir}"
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
  echo "  --download-rir <download_rir> # whether to download or simulate RIRs (Default=${download_rir})"
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


sph2pipe=sph2pipe
if ! command -v "${sph2pipe}" &> /dev/null; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

if ! command -v "mpiexec" &> /dev/null; then
  echo "Could not find (or execute) the mpiexec program";
  exit 1;
fi


if [[ ! -d ${wsj_zeromean_wav} ]]; then
  echo "Creating zero-mean normalized wsj at '${wsj_zeromean_wav}'."
  ${train_cmd} mpiexec -np ${nj} python -m sms_wsj.database.wsj.write_wav \
      with dst_dir=${wsj_zeromean_wav} wsj0_root=${wsj0_path} \
      wsj1_root=${wsj1_path} sample_rate=${sample_rate}
fi


mkdir -p ${json_dir}
if [[ ! -f ${json_dir}/wsj_${sample_rate}_zeromean.json ]]; then
	echo "Creating ${json_dir}/wsj_${sample_rate}_zeromean.json"
	${train_cmd} mpiexec -np ${nj} python -m sms_wsj.database.wsj.create_json \
		with json_path=${json_dir}/wsj_${sample_rate}_zeromean.json \
    database_dir=${wsj_zeromean_wav} as_wav=True
fi


if [[ ! -d ${rir_dir} ]]; then
  mkdir -p ${rir_dir}
  if ${download_rir}; then
    echo "Downloading RIRs (50.8 GB) in '${rir_dir}'"
    wget -qO- https://zenodo.org/record/3517889/files/sms_wsj.tar.gz.parta{a,b,c,d,e} \
      | tar -C ${rir_dir}/ -zx --checkpoint=10000 --checkpoint-action=echo="%u/5530000 %c"
  else
    echo "Simulating RIRs in '${rir_dir}'"
    # This takes around 1900 / (ncpus - 1) hours.
    ${train_cmd} mpiexec -np ${nj} python -m sms_wsj.database.create_rirs \
      with database_path=${rir_dir}
  fi
fi


if [[ ! -f ${json_dir}/sms_wsj.json ]]; then
	echo "Creating ${json_dir}/sms_wsj.json"
	${train_cmd} mpiexec -np ${nj} python -m sms_wsj.database.create_json \
		with json_path=${json_dir}/sms_wsj.json rir_dir=${rir_dir} \
		wsj_json_path=${json_dir}/wsj_${sample_rate}_zeromean.json
fi


echo "Creating ${sms_wsj_wav} audio data in '${sms_wsj_wav}'"
${train_cmd} mpiexec -np ${nj} python -m sms_wsj.database.write_files \
	with dst_dir=${sms_wsj_wav} json_path=${json_dir}/sms_wsj.json \
	write_all=True new_json_path=${json_dir}/sms_wsj.json


# The total disc usage of SMS-WSJ is 442.1 GiB.
# directory	      disc usage
# --------------------------
# tail	          120.1 GiB
# early	          120.1 GiB
# observation	    60.0 GiB
# noise	          60.0 GiB
# rirs	          52.6 GiB
# wsj_8k_zeromean	29.2 GiB
# sms_wsj.json	  1397 MiB
# wsj_8k.json	    316 MiB
