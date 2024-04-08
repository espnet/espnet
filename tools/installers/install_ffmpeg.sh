#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

unames="$(uname -s)"
unamem="$(uname -m)"

dirname=ffmpeg-release
rm -rf ${dirname}

if [ -x "$(command -v ffmpeg)" ]; then
    echo "ffmpeg is already installed in system"
    exit 0
fi

if [[ ${unames} =~ Linux ]]; then
    if [ "${unamem}" = x86_64 ]; then
        unamem=amd64
    fi
    url="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-${unamem}-static.tar.xz"
    wget --no-check-certificate --trust-server-names "${url}"
    tar xvf "ffmpeg-release-${unamem}-static.tar.xz"
    ffmpegdir="$(ls -d ffmpeg-*-static)"
    ln -sf "${ffmpegdir}" "${dirname}"
elif [[ ${unames} =~ Darwin ]]; then
    # bins="ffmpeg ffprobe ffplay ffserver"
    bins="ffmpeg ffprobe ffplay"
    for bin in ${bins}; do
        url="https://evermeet.cx/ffmpeg/getrelease/${bin}/zip"
        wget --no-check-certificate --trust-server-names "${url}" -O "${bin}-release.zip"
        unzip -o "${bin}-*.zip" -d ${dirname}
    done
elif [[ ${unames} =~ MINGW || ${unames} =~ CYGWIN || ${unames} =~ MSYS ]]; then
    # Windows
    url=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
    wget --no-check-certificate --trust-server-names "${url}" -O "ffmpeg-release-essentials_build.zip"
    unzip -o ffmpeg-release-essentials_build.zip
    ffmpegdir="$(ls -d ffmpeg-*-essentials_build)"
    ln -sf "${ffmpegdir}"/bin "${dirname}"
else
    echo "$0: Warning: not supported platform: ${unames}"
fi
