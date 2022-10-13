#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

unames="$(uname -s)"
if [[ ! ${unames} =~ Linux && ! ${unames} =~ Darwin ]]; then
    echo "Warning: This script may not work with ${unames}. Exit with doing nothing"
    exit 0
fi


rm -rf nkf
mkdir -p nkf
(
    set -euo pipefail
    cd nkf
    wget --tries=3 https://ja.osdn.net/dl/nkf/nkf-2.1.4.tar.gz
    tar zxvf nkf-2.1.4.tar.gz
    (
        set -euo pipefail
        cd nkf-2.1.4
        make prefix=.
    )
)
