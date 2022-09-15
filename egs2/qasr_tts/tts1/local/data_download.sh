#!/usr/bin/env bash

# Copyright 2021 Massa Baali
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail
echo $download_dir
cwd=$(pwd)

if [ ! -d "${download_dir}/qasr_tts-1.0" ] && [ -f "qasr_tts-1.0.zip" ]; then
    mkdir -p "${download_dir}"
    cd "${download_dir}"
    unzip "${cwd}/qasr_tts-1.0"
    cd "${cwd}"
    echo "successfully prepared data."
    
else
    echo "Go to this link https://arabicspeech.org/qasr_tts "
    exit 1
fi

exit 0 