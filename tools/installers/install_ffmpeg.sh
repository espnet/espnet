#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

unames="$(uname -s)"
unamem="$(uname -m)"

dirname=ffmpeg-release

if [ -x "$(command -v ffmpeg)" ]; then
    echo "ffmpeg is already installed in system"
    exit 0
fi

if [ -x "${dirname}/ffmpeg" ]; then
    echo "ffmpeg is already installed in ${dirname}"
    exit 0
fi

existing_ffmpegdir="$(find . -maxdepth 1 -type d -name 'ffmpeg-*-static' -print -quit | sed 's|^\./||')"
if [ -n "${existing_ffmpegdir}" ] && [ -x "${existing_ffmpegdir}/ffmpeg" ]; then
    ln -sfn "${existing_ffmpegdir}" "${dirname}"
    echo "ffmpeg is already installed in ${existing_ffmpegdir}"
    exit 0
fi

download_archive() {
    local url="$1"
    local output="$2"
    local tmp_output="${output}.tmp"

    rm -f "${tmp_output}"
    wget --no-check-certificate --tries=5 --waitretry=5 --retry-connrefused \
        --timeout=30 --read-timeout=30 --trust-server-names -O "${tmp_output}" "${url}"
    mv "${tmp_output}" "${output}"
}

if [[ ${unames} =~ Linux ]]; then
    if [ "${unamem}" = x86_64 ]; then
        unamem=amd64
    fi
    ffmpeg_name="ffmpeg-release-${unamem}-static.tar.xz"
    PRIMARY_URL="https://johnvansickle.com/ffmpeg/releases/${ffmpeg_name}"
    BACKUP_URL="https://huggingface.co/espnet/ci_tools/resolve/main/${ffmpeg_name}"
    if [ -f "${ffmpeg_name}" ]; then
        echo "Using cached archive ${ffmpeg_name}"
    else
        if ! download_archive "${PRIMARY_URL}" "${ffmpeg_name}"; then
            echo "Primary download failed, trying backup URL..."
            if ! download_archive "${BACKUP_URL}" "${ffmpeg_name}"; then
                echo "Both primary and backup downloads failed"
                exit 1
            fi
        fi
    fi
    rm -rf "${dirname}" ffmpeg-*-static
    tar xvf "${ffmpeg_name}"
    ffmpegdir="$(find . -maxdepth 1 -type d -name 'ffmpeg-*-static' -print -quit | sed 's|^\./||')"
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
