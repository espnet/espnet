#!/usr/bin/env bash
# Stage-3 equivalent (format_wav_scp) run PER corpus-language dir.
#
# Deliberately not run on the combined dir: doing so would mean re-processing
# every existing file each time a new corpus is added. Per-dset outputs are
# combined afterwards, so new data only pays for itself.
#
# Uses --count-only: this audio is 16 kHz mono FLAC with no segments, so
# format_wav_scp references it in place and the decoded samples are discarded.
# Reading the header instead is ~4x faster and verified byte-identical.
set -e
set -u
set -o pipefail
log() { echo "[$(date '+%F %T')] $*"; }

. ./cmd.sh
. ./path.sh

nj=64
fs=16k
audio_format=flac
dst_prefix=pretrain
data_feats=dump/raw
count_only=true

. utils/parse_options.sh

mapfile -t DSETS < <(ls -d data/${dst_prefix}_*/ 2>/dev/null \
                     | sed 's|^data/||; s|/$||' | grep -v "_all$" | sort)
[ ${#DSETS[@]} -gt 0 ] || { log "ERROR: no data/${dst_prefix}_* dirs"; exit 1; }
log "formatting ${#DSETS[@]} datasets with nj=${nj}, count_only=${count_only}"

for dset in "${DSETS[@]}"; do
    out="${data_feats}/org/${dset}"
    if [ -f "${out}/.done" ]; then
        log "skip ${dset} (already done)"; continue
    fi
    n=$(wc -l < "data/${dset}/wav.scp")
    log "=== ${dset}: ${n} utterances -> ${out}"
    rm -rf "${out}"
    utils/copy_data_dir.sh --validate_opts --non-print "data/${dset}" "${out}"
    scripts/audio/format_wav_scp.sh \
        --nj "${nj}" --cmd "${train_cmd}" \
        --audio-format "${audio_format}" --fs "${fs}" \
        --count-only "${count_only}" \
        "data/${dset}/wav.scp" "${out}"
    echo "raw" > "${out}/feats_type"
    got=$(wc -l < "${out}/utt2num_samples")
    [ "${got}" -eq "${n}" ] || { log "ERROR: ${dset} produced ${got}/${n} utt2num_samples"; exit 1; }
    touch "${out}/.done"
    log "  ${dset}: ${got} utterances formatted"
done
log "all datasets formatted"
