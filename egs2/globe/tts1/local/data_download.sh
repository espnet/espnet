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

# 2️⃣  Skip if already done
if [ -f "${db_root}/.complete" ]; then
    echo "GLOBE already downloaded.  Skipping."
    exit 0
fi

mkdir -p "${db_root}"
cd       "${db_root}"

echo "Downloading GLOBE V2 (≈76 GB)…"
echo "  – This uses git-lfs; make sure it’s installed."

# 3️⃣  Clone the dataset repo (shallow, Git-LFS pulls large audio files)
if ! command -v git-lfs &>/dev/null; then
    echo "ERROR: git-lfs not found.  Install it with conda, apt, or brew."
    exit 1
fi
git lfs install                       # no-op if already done
git clone --depth 1 \
          https://huggingface.co/datasets/MushanW/GLOBE_V2 globe_v2

# 4️⃣  Mark completion
touch "${db_root}/.complete"
echo "Successfully downloaded GLOBE V2."

cd - >/dev/null
