#!/usr/bin/env bash

# Copyright 2023 ISSAI (author: Yerbolat Khassanov)
# Apache 2.0

KSC=$1
cd "${KSC}"

# Kazakh speech corpus (KSC):
if [ ! -e ISSAI_KSC_335RS_v1.1_flac ]; then
  echo "$0: downloading KSC data (it won't re-download if it was already downloaded.)"
  wget --continue https://www.openslr.org/resources/102/ISSAI_KSC_335RS_v1.1_flac.tar.gz || exit 1
  tar xf "ISSAI_KSC_335RS_v1.1_flac.tar.gz"
else
    echo "$0: not downloading or un-tarring ISSAI_KSC_335RS_v1.1_flac because it already exists."
fi


exit 0
