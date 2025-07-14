#!/usr/bin/env bash
# Download and prepare GLOBE-V2 audio for Kaldi
# Usage: local/data_download.sh <db_root>
#  1) clone/pull all .flac via git-lfs
#  2) convert them in parallel to 24 kHz WAV

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <db_root>"
  exit 1
fi
db_root=$1
repo_dir="$db_root/globe_v2"
wav_root="$repo_dir/data"   # 原始 flac 存在 data/.../*.flac
nj=32                       # 并行 job 数，可调整

mkdir -p "$db_root"

# 1) 如果已经下载完成，跳过
if [ -f "$db_root/.complete" ]; then
  echo "✔️  GLOBE already downloaded. Skipping."
  exit 0
fi

# 2) 如果已经 clone 过，只拉 .flac
if [ -d "$repo_dir/.git" ]; then
  echo "▶️  Resuming GLOBE_V2 clone → pulling .flac only"
  cd "$repo_dir"
  git lfs install --local
  git lfs pull --include="*.flac" --exclude=""
  cd - >/dev/null
else
  # fresh shallow clone
  echo "▶️  Cloning GLOBE_V2 (shallow)…"
  cd "$db_root"
  git clone --depth 1 https://huggingface.co/datasets/MushanW/GLOBE_V2 globe_v2
  cd - >/dev/null

  echo "▶️  Pulling down all .flac (via Git-LFS)…"
  cd "$repo_dir"
  git lfs install
  git lfs pull --include="*.flac" --exclude=""
  cd - >/dev/null
fi

# 3) multithread 转成 WAV （24 kHz，单通道）
echo "▶️  Converting all .flac → .wav with $nj jobs…"
command -v parallel >/dev/null 2>&1 || {
  echo "ERROR: requires GNU parallel, please install it"; exit 1;
}
command -v sox >/dev/null 2>&1 || {
  echo "ERROR: requires sox, please install it"; exit 1;
}

# find all flac file
find "$wav_root" -type f -name '*.flac' | \
  parallel -j "$nj" --bar '
    f={};
    out="${f%.flac}.wav";
    # skip if complete
    [ -f "$out" ] && continue;
    # sox
    sox "$f" -c 1 -r 24000 "$out"
  '

# 4) mark as complete
touch "$db_root/.complete"
echo "✔️  All done. WAV files are ready alongside FLACs."
