#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

lang=
release_name="SPRING_INX"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

stage=1
stop_stage=100000

lang=${lang^}
echo Language : ${lang}

data_url="https://asr.iitm.ac.in/SPRING_INX/data/${release_name}_${lang}"

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${release_name}" ]; then
    log "Fill the value of '${release_name}' of db.sh"
    exit 1
fi

global_version=0
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
	echo "stage 1: Data Download to '${!release_name}'"
	mkdir -p ${!release_name}
	version=1
	while [ $version -ge 1 ]; do
	    if ! local/download_and_untar.sh --remove-archive ${!release_name} ${data_url}_R${version} ${release_name} ${lang} ${version}; then
	        version=0
	        break
	    else
	       global_version=${version}
	    fi
		if [ $version -eq  0 ]; then
			break
		else
			version=$(expr $version + 1)
		fi
	done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"

    mkdir -p data


    for release in $(seq 1 ${global_version}); do
    	for directory in dev eval train; do

			sed -i "s:SPRING_INX/::g" ${!release_name}/${release_name}_${lang}_R${release}/${directory}/wav.scp
			sed -i "s://:/:g" ${!release_name}/${release_name}_${lang}_R${release}/${directory}/wav.scp

    		if [ -d data/${directory}_R${release} ]; then
    			rm -r data/${directory}_R${release}

    			cp -r ${!release_name}/${release_name}_${lang}_R${release}/${directory} data/${directory}_R${release}
	    	else

	    		cp -r ${!release_name}/${release_name}_${lang}_R${release}/${directory} data/${directory}_R${release}
    		fi
    		echo "
Copied ${directory}_R${release} to the data folder."

			./utils/validate_data_dir.sh  data/${directory}_R${release}  --no-feats

    	done
    done

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "
stage 3:
Combining train, dev and eval from all releases."

    for part in dev eval train; do
    	if [[ ${global_version} -eq 1 ]]; then
    		if [[ -d data/${part} ]]; then
    			rm -r data/${part}
    		fi
    		mv data/${part}_R1 data/${part}
    	else
			data_releases=""
			for release in $(seq 1 ${global_version});do
				data_releases="${data_releases} data/${part}_R${release}"
			done
			bash utils/combine_data.sh data/${part} ${data_releases} >/dev/null

			echo "
	Combined all ${part} versions."

			./utils/validate_data_dir.sh  data/${directory}_R${release} --no-feats
		fi
    done
fi

echo "
Successfully finished. [elapsed=${SECONDS}s]"
