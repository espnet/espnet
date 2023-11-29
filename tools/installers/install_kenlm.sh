#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

boost_version=1.81.0

if [ ! -d boost_${boost_version//./_} ]; then
    if [ ! -e "boost_"${boost_version//./_}".tar.bz2" ]; then
        wget --no-check-certificate https://boostorg.jfrog.io/artifactory/main/release/"${boost_version}"/source/boost_"${boost_version//./_}".tar.bz2
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
