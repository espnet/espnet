#!/usr/bin/env bash

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

gdrive_url="https://drive.google.com/drive/folders/1WRt-EprWs-2rmYxoWYT9_13omlhDHcaL"

cwd=$(pwd)

if [ ! -e "${download_dir}/dailytalk" ] && [ ! -e "${download_dir}/DailyTalk" ]; then
    mkdir -p "${download_dir}"
    cd "${download_dir}"

    if ! command -v gdown > /dev/null 2>&1; then
        echo "Error: gdown is not installed."
        echo "Please install it with: pip install gdown"
        exit 1
    fi

    echo "Downloading DailyTalk from Google Drive..."
    gdown --folder "${gdrive_url}"

    if [ ! -f dailytalk.zip ]; then
        zip_file=$(find . -maxdepth 2 -name "dailytalk.zip" -o -name "DailyTalk.zip" | head -n 1)
        if [ -z "${zip_file}" ]; then
            echo "Error: Cannot find dailytalk.zip after download."
            exit 1
        fi
    else
        zip_file="dailytalk.zip"
    fi

    echo "Extracting ${zip_file}..."
    unzip "${zip_file}"

    cd "${cwd}"
    echo "successfully prepared data."
else
    echo "already exists. skipped."
fi
