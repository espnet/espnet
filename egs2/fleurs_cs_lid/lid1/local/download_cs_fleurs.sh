#!/usr/bin/env bash
# Clone the byan/cs-fleurs dataset at the pinned revision (git-lfs required).
set -e
set -u
set -o pipefail

root=downloads/cs-fleurs
revision=0cdbf166c5517ae4b6eb1c54248522eedec53017

. utils/parse_options.sh

if [ ! -d "${root}/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/byan/cs-fleurs "${root}"
fi
git -C "${root}" fetch origin "${revision}"
GIT_LFS_SKIP_SMUDGE=1 git -C "${root}" checkout --detach "${revision}"
git -C "${root}" lfs pull

for subset in xtts/train read/test xtts/test1 xtts/test2 mms/test; do
    if [ ! -s "${root}/${subset}/metadata.jsonl" ]; then
        echo "Error: missing ${root}/${subset}/metadata.jsonl" >&2
        exit 1
    fi
done
