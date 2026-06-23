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
help_message=$(cat << EOF
Usage: $0

Options:
    --remove_archive (bool): true or false
      With remove_archive=True, the archives will be removed after being successfully downloaded and un-tarred.
EOF
)
SECONDS=0

# Data preparation related
# data_url=https://openslr.trmal.net/resources/123
data_url=https://www.openslr.org/resources/123/
tar_name=MagicData-RAMC.tar.gz
remove_archive=false

# Filtering related (forwarded to local/prepare_data.py).
# All min/max filters and drop_special_segments are train-only; dev/test are
# kept as released so reported metrics reflect the full evaluation distribution.
# Paralinguistic tags ([+]/[*]/[LAUGHTER]/[SONANT]/[MUSIC]) are preserved as
# atomic tokens (see data/nlsyms.txt written below).
drop_special_segments=false
min_time=300       # minimum segment duration (ms)
max_time=30000     # maximum segment duration (ms)
min_text=1         # minimum cleaned-text character count
max_text=200       # maximum cleaned-text character count

log "$0 $*"


. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if [ -z "${MAGICDATA_RAMC}" ]; then
  log "Error: \$MAGICDATA_RAMC is not set in db.sh."
  exit 2
fi

if [ ! -d "${MAGICDATA_RAMC}" ]; then
  mkdir -p "${MAGICDATA_RAMC}"
fi
MAGICDATA_RAMC=$(cd ${MAGICDATA_RAMC}; pwd)

# Download and extract data
if [ -d "${MAGICDATA_RAMC}/MDT2021S003" ]; then
  log "Data directory ${MAGICDATA_RAMC}/MDT2021S003 already exists. Skipping data download and extraction."
else
  if [ -f "${MAGICDATA_RAMC}/${tar_name}" ]; then
    log "Archive ${MAGICDATA_RAMC}/${tar_name} already exists. Skipping download."
  else
    log "Download data to ${MAGICDATA_RAMC}..."
    full_url="${data_url}/${tar_name}"
    if ! wget -O "${MAGICDATA_RAMC}/${tar_name}" "${full_url}"; then
      echo "$0: error executing wget ${full_url}"
      rm -f "${MAGICDATA_RAMC}/${tar_name}"
      exit 1
    fi
  fi
  log "Extracting data from ${MAGICDATA_RAMC}/${tar_name}..."
  if ! tar -xzf "${MAGICDATA_RAMC}/${tar_name}" -C "${MAGICDATA_RAMC}"; then
    echo "$0: error executing tar -xzf ${MAGICDATA_RAMC}/${tar_name} -C ${MAGICDATA_RAMC}"
    exit 1
  fi
  if [ "${remove_archive}" = "true" ]; then
    log "Removing archive ${MAGICDATA_RAMC}/${tar_name}"
    rm -f "${MAGICDATA_RAMC}/${tar_name}"
  fi
fi

# Convert data to Kaldi-style data directories under data/{train,dev,test}
output_dir=data
mkdir -p "${output_dir}"

python_opts=""
if [ "${drop_special_segments}" = "true" ]; then
  python_opts="${python_opts} --drop-special-segments"
fi

log "Preparing Kaldi-style data directories under ${output_dir}/"
python3 local/prepare_data.py \
  --raw-data-dir "${MAGICDATA_RAMC}" \
  --output-dir "${output_dir}" \
  --filter-min-time ${min_time} \
  --filter-max-time ${max_time} \
  --filter-min-text ${min_text} \
  --filter-max-text ${max_text} \
  ${python_opts}

# Validate and clean each split:
#   fix_data_dir.sh sorts, drops utts missing in some file, regenerates spk2utt
#   validate_data_dir.sh --no-feats checks the five files are mutually consistent
for split in train dev test; do
  utils/fix_data_dir.sh "${output_dir}/${split}"
  utils/validate_data_dir.sh --no-feats --non-print "${output_dir}/${split}"
done

# Emit the non-linguistic-symbol list. run.sh passes this via --nlsyms_txt so
# the stage-5 token-list builder keeps these paralinguistic tags as atomic
# tokens (rather than splitting them into individual characters).
nlsyms=${output_dir}/nlsyms.txt
log "Writing non-linguistic symbols to ${nlsyms}"
printf '%s\n' '[+]' '[*]' '[LAUGHTER]' '[SONANT]' '[MUSIC]' > "${nlsyms}"

log "Successfully finished. [elapsed=${SECONDS}s]"
