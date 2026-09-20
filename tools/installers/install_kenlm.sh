#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=tools/installers/download_with_retry.sh
. "$(dirname "$0")"/download_with_retry.sh

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

boost_version=1.81.0

if [ ! -d boost_${boost_version//./_} ]; then
    if [ ! -e "boost_"${boost_version//./_}".tar.bz2" ]; then
        download_with_retry \
            https://sourceforge.net/projects/boost/files/boost/"${boost_version}"/boost_"${boost_version//./_}".tar.bz2 \
            boost_"${boost_version//./_}".tar.bz2 --no-check-certificate
    fi
    tar xvf boost_"${boost_version//./_}".tar.bz2
fi

if [ ! -d boost_${boost_version//./_}_build ]; then
    (
        set -euo pipefail
        cd boost_${boost_version//./_}
        ./bootstrap.sh
        ./b2 install --prefix=$(pwd)/../boost_${boost_version//./_}_build install
    )
fi

if [ ! -d kenlm ]; then
    git clone https://github.com/kpu/kenlm.git
fi

(
    set -euo pipefail
    cd kenlm

    mkdir -p build
    (
        set -euo pipefail
        cd build && cmake -DCMAKE_PREFIX_PATH=$(pwd)/../../boost_${boost_version//./_}_build .. && make
    )
    (
        set -euo pipefail
        python3 -m pip install -e .
    )
)
