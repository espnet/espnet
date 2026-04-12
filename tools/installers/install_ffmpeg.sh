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
    ffmpeg_name="ffmpeg-release-${unamem}-static.tar.xz"
    PRIMARY_URL="https://johnvansickle.com/ffmpeg/releases/${ffmpeg_name}"
    BACKUP_URL="https://huggingface.co/espnet/ci_tools/resolve/main/${ffmpeg_name}"
    if ! wget --no-check-certificate --tries=3 --trust-server-names -O "${ffmpeg_name}" "${PRIMARY_URL}"; then
        echo "Primary download failed, trying backup URL..."
        if ! wget --no-check-certificate --tries=3 -O "${ffmpeg_name}" "${BACKUP_URL}"; then
            echo "Both primary and backup downloads failed"
            exit 1
        fi
    fi
    tar xvf "${ffmpeg_name}"
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
