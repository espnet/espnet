#!/usr/bin/env bash
set -euo pipefail
[ -f ./path.sh ] && . ./path.sh


if ! command conda  &>/dev/null; then
  echo "Conda command not found, please follow the instructions on
  this recipe README.md on how to install ESPNet with conda as the venv."
fi

# install lhotse from master, we need the most up-to-date one
pip install git+https://github.com/lhotse-speech/lhotse

#check if kaldi has been installed and compiled
if ! command -v wav-reverberate &>/dev/null; then
  echo "It seems that wav-reverberate Kaldi command cannot be found.
  This happens if you don't have compiled and installed Kaldi.
  Please follow instructions in ${MAIN_ROOT}/tools/kaldi/INSTALL. "
fi

# install s3prl
${MAIN_ROOT}/tools/installers/install_s3prl.sh

if ! command -v gss &>/dev/null; then
  conda install -yc conda-forge cupy=10.2
  ${MAIN_ROOT}/tools/installers/install_gss.sh
fi

sox_conda=`command -v $(dirname $(which python))/sox 2>/dev/null`
if [ -z "${sox_conda}" ]; then
  echo "install conda sox (v14.4.2)"
  conda install -c conda-forge sox
fi

ffmpeg=`command -v ffmpeg 2>/dev/null` \
  || { echo  >&2 "ffmpeg not found on PATH. Please install it manually (you will need version 4 and higher)."; exit 1; }

# If sox is found on path, check if the version is correct
if [ ! -z "$ffmpeg" ]; then
  ffmpeg_version=`$ffmpeg -version 2>&1| head -1`
  ffmpeg_version=$(cut -d'_' -f2 <<<"$ffmpeg_version")
  if [[ ! $ffmpeg_version =~ 4.* ]]; then
    echo "Unsupported ffmpeg version $ffmpeg_version found on path. You will need version 4 and higher."
    exit 1
  fi
fi