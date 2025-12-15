#!/usr/bin/env bash
# Usage: bash local/sort_spk_embed_scp.sh <dumpdir> <spk_embed_tag>
# Example: bash local/sort_spk_embed_scp.sh dump/44k espnet_spk
set -euo pipefail

dumpdir=${1:?need dumpdir}
tag=${2:?need spk_embed_tag}

# Function to sort all standard Kaldi data files in a folder
sort_data_dir() {
  local dir=$1
  for f in text wav.scp utt2spk spk2utt utt2dur spk2gender; do
    [ -f "${dir}/${f}" ] && LC_ALL=C sort -k1,1 "${dir}/${f}" -o "${dir}/${f}"
  done
}

# Sort all standard data files for each dataset directory found
for datadir in "${dumpdir}/raw/org/"* "${dumpdir}/raw/"*; do
  [ -d "$datadir" ] || continue
  echo "[info] Sorting data files in $datadir"
  sort_data_dir "$datadir"
done

# Now sort the embedding scp files
for scp in "${dumpdir}/${tag}"/*/"${tag}.scp"; do
  dset=$(basename "$(dirname "$scp")")

  # Train/val live under raw/org, tests under raw/
  if [[ "$dset" == "train" || "$dset" == "val" ]]; then
    rawdir="${dumpdir}/raw/org/${dset}"
  else
    rawdir="${dumpdir}/raw/${dset}"
  fi

  ids="${rawdir}/text"
  [[ -f "$ids" ]] || { echo "Missing $ids"; exit 1; }

  echo "[info] Filtering & sorting ${scp} to match ${ids}"
  tmp="${scp}.tmp"
  awk '{print $1}' "$ids" > "${scp}.ids"
  utils/filter_scp.pl -f 1 "${scp}.ids" "${scp}" | LC_ALL=C sort -k1,1 > "$tmp"
  mv "$tmp" "$scp"

  # Quick sanity check: key order match
  paste <(awk '{print $1}' "$ids") <(awk '{print $1}' "$scp") \
    | awk '{ if ($1!=$2) { print "mismatch at line", NR, $0; exit 1 } }'
done
echo "[ok] All ${tag}.scp files and data folders are sorted/aligned."
