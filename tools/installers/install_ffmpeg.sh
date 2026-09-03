#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

unames="$(uname -s)"
unamem="$(uname -m)"

# shellcheck source=tools/installers/download_with_retry.sh
. "$(dirname "$0")"/download_with_retry.sh

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
        if [ "$(id -u)" = "0" ]; then
            apt_install_cmd="apt-get update -qq && apt-get install -qq -y ffmpeg"
        elif command -v sudo > /dev/null 2>&1; then
            apt_install_cmd="sudo -n apt-get update -qq && sudo -n apt-get install -qq -y ffmpeg"
        else
            apt_install_cmd=""
            echo "Neither root nor sudo is available; falling back to direct download..."
        fi

        if [ -n "${apt_install_cmd}" ] && eval "${apt_install_cmd}"; then
            if command -v ffmpeg > /dev/null 2>&1; then
                echo "ffmpeg installed successfully via apt-get"
                exit 0
            fi
        fi
        echo "apt-based ffmpeg setup failed or ffmpeg was not found after install, falling back to download..."
    fi

    if [ "${unamem}" = x86_64 ]; then
        unamem=amd64
    fi
    ffmpeg_name="ffmpeg-release-${unamem}-static.tar.xz"
    PRIMARY_URL="https://johnvansickle.com/ffmpeg/releases/${ffmpeg_name}"
    BACKUP_URL="https://huggingface.co/espnet/ci_tools/resolve/main/${ffmpeg_name}"


    if ! download_with_retry "${PRIMARY_URL}" "${ffmpeg_name}" \
            --no-check-certificate; then
        echo "Primary download failed, trying backup URL..."
        wget_args=()
        if [ -n "${HF_TOKEN:-}" ]; then
            wget_args+=("--header=Authorization: Bearer ${HF_TOKEN}")
        else
            echo "HF_TOKEN is not set, backup download may fail if the file has many downloads"
        fi
        if ! download_with_retry "${BACKUP_URL}" "${ffmpeg_name}" \
                --no-check-certificate "${wget_args[@]}"; then
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
