#!/usr/bin/env bash

set -euo pipefail

# install kaldi
[ ! -d tools/kaldi ] && git clone https://github.com/kaldi-asr/kaldi --depth 1 tools/kaldi
(
    cd ./tools/kaldi/tools || exit 1
    echo "" > extras/check_dependencies.sh
    chmod +x extras/check_dependencies.sh
)

# download pre-built kaldi binary
# TODO(karita) support non ubuntu env
#
# `wget --tries=3` was not a retry for the way this actually fails. GitHub's
# release assets redirect to release-assets.githubusercontent.com, and the
# handshake there is what breaks:
#
#   Unable to establish SSL connection.
#   ##[error]Process completed with exit code 4.
#
# wget's --tries does not start a new connection for that, so the download failed
# on the first attempt and reported three. The shared helper uses --tries=1 with
# backoff instead, and checks the result really is an archive.
# shellcheck source=tools/installers/download_with_retry.sh
. ./tools/installers/download_with_retry.sh

if [ ! -e ubuntu16-featbin.tar.gz ]; then
    download_with_retry \
        https://github.com/espnet/kaldi-bin/releases/download/v0.0.1/ubuntu16-featbin.tar.gz \
        ubuntu16-featbin.tar.gz
fi
tar -xf ./ubuntu16-featbin.tar.gz
cp featbin/* tools/kaldi/src/featbin/
