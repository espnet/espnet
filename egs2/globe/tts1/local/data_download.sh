#!/usr/bin/env bash
# Download and unpack the GLOBE V2 corpus
# Usage: local/download.sh <db_root>

set -euo pipefail

# 1️⃣  Check args
if [ $# -ne 1 ]; then
    echo "Usage: $0 <db_root>"
    exit 1
fi
db_root=$1
mkdir -p "${db_root}"

if [ -f "${db_root}/.complete" ]; then
  echo "GLOBE already downloaded. Skipping."
  exit 0
fi

# 2️⃣ Resume: if the repo is there, just pull missing Parquet shards
if [ -d "${db_root}/globe_v2/.git" ]; then
    echo "Found existing globe_v2 clone – finishing Git-LFS checkout."
    cd "${db_root}/globe_v2"
    git lfs install --local
    git lfs pull --include "*.parquet" --exclude ""
    cd - >/dev/null
    touch "${db_root}/.complete"
    echo "✔️  GLOBE V2 resume complete."
    exit 0
fi



cd       "${db_root}"

echo "Downloading GLOBE V2 (≈76 GB)…"
echo "  – This uses git-lfs; make sure it’s installed."

# 3️⃣  Clone the dataset repo (shallow, Git-LFS pulls large audio files)
if ! command -v git-lfs &>/dev/null; then
    echo "ERROR: git-lfs not found.  Install it with conda, apt, or brew."
    exit
fi
git lfs install                       # no-op if already done
git clone --depth 1 \
          https://huggingface.co/datasets/MushanW/GLOBE_V2 globe_v2

# 4️⃣  Mark completion
touch "${db_root}/.complete"
echo "Successfully downloaded GLOBE V2."

cd - >/dev/null
