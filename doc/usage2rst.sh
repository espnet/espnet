#!/usr/bin/env bash

set -euo pipefail

if [ $1 == "--help" ]; then
    echo "Usage: $0 <shell-script>"
    exit 0;
fi

real=$(realpath $1)
githash=$(git rev-parse HEAD)

cd ./egs2/wsj/asr1
. path.sh

cmd=$(basename $1)
len=${#cmd}
r=$(dirname $real)
sep=$(printf '~%.0s' $(seq $len))
usage=$($real --help |& sed "s?${r}/??g" | grep -v -e '--help' | sed "s/^/    /g")
sourceurl="https://github.com/espnet/espnet/blob/${githash}\/$1"
cat <<EOF
.. _${cmd}

${cmd}
${sep}

\`source <${sourceurl}>\`_

.. code-block:: none

${usage}
EOF
