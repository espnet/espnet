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
    # Try system package manager first to avoid hitting rate-limited download servers
    # (especially important in CI where many jobs may run concurrently)
    if command -v apt-get > /dev/null 2>&1; then
        echo "Trying to install ffmpeg via apt-get..."
        if { sudo -n apt-get update -qq && sudo -n apt-get install -qq -y ffmpeg; } \
                || apt-get install -qq -y ffmpeg; then
            if command -v ffmpeg > /dev/null 2>&1; then
                echo "ffmpeg installed successfully via apt-get"
                exit 0
            fi
        fi
        echo "apt-get install failed or ffmpeg not found after install, falling back to download..."
    fi

    if [ "${unamem}" = x86_64 ]; then
        unamem=amd64
    fi
    ffmpeg_name="ffmpeg-release-${unamem}-static.tar.xz"
    PRIMARY_URL="https://johnvansickle.com/ffmpeg/releases/${ffmpeg_name}"
    BACKUP_URL="https://huggingface.co/espnet/ci_tools/resolve/main/${ffmpeg_name}"

    download_with_retry() {
        local url="$1"
        local output="$2"
        local max_attempts=3
        local attempt=1
        local wait=5
        while [ "${attempt}" -le "${max_attempts}" ]; do
            if wget --no-check-certificate --trust-server-names --tries=1 -O "${output}" "${url}"; then
                return 0
            fi
            echo "Attempt ${attempt}/${max_attempts} failed for ${url}"
            if [ "${attempt}" -lt "${max_attempts}" ]; then
                echo "Waiting ${wait}s before retry..."
                sleep ${wait}
                wait=$((wait * 2))
            fi
            attempt=$((attempt + 1))
        done
        return 1
    }

    if ! download_with_retry "${PRIMARY_URL}" "${ffmpeg_name}"; then
        echo "Primary download failed, trying backup URL..."
        if ! download_with_retry "${BACKUP_URL}" "${ffmpeg_name}"; then
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
