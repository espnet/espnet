#!/usr/bin/env bash
# Download the byan/cs-yodas metadata and audio archives at the pinned revision
# and extract only the wav files of the base-language + English records.
set -e
set -u
set -o pipefail

root=downloads/cs-yodas
revision=e51028041b403f63c99ac91a4af040e72d0cad0e

. utils/parse_options.sh

base_url=https://huggingface.co/datasets/byan/cs-yodas/resolve/${revision}
mkdir -p "${root}/metadata" "${root}/archives" "${root}/audio"

download() {
    if [ ! -s "${root}/$1" ]; then
        curl -fL -sS --retry 5 --retry-delay 10 "${base_url}/$2" -o "${root}/$1.part"
        mv "${root}/$1.part" "${root}/$1"
    fi
}

for lang in ara cmn fra hin jpn rus; do
    download "metadata/${lang}.jsonl" "${lang}.jsonl"
done
for name in ara.tar cmn.tar hin.tar jpn.tar rus.tar fra.tar.aa fra.tar.ab; do
    download "archives/${name}" "${name}"
done

# Select the wav paths of the records used by the recipe
python3 local/list_cs_yodas_wavs.py "${root}"

for lang in ara cmn hin jpn rus; do
    tar -xf "${root}/archives/${lang}.tar" -C "${root}/audio" -T "${root}/${lang}.paths"
done
cat "${root}/archives/fra.tar.aa" "${root}/archives/fra.tar.ab" \
    | tar -xf - -C "${root}/audio" -T "${root}/fra.paths"
