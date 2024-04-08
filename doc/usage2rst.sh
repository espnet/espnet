#!/usr/bin/env bash

set -euo pipefail

if [ $1 == "--help" ]; then
    echo "Usage: $0 <shell-script>"
    exit 0;
fi

real=$(realpath $1)

cd ./egs2/wsj/asr1
. path.sh

cmd=$(basename $1)
len=${#cmd}
r=$(dirname $real)
sep=$(printf '~%.0s' $(seq $len))
usage=$($real --help |& sed "s?${r}/??g" | grep -v -e '--help' | sed "s/^/    /g")
cat <<EOF
.. _${cmd}:

${cmd}
${sep}

.. code-block:: none

${usage}
EOF
