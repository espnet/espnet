#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
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
