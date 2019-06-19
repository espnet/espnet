#!/usr/bin/env bash

set -euo pipefail

cd ./egs/wsj/asr1
. path.sh

cmd=$(basename $1)
len=${#cmd}
r=$(pwd)/../../../utils/
sep=$(printf '~%.0s' $(seq $len))
usage=$($cmd --help |& sed "s?${r}??g" | grep -v -e '--help' | sed "s/^/    /g")
cat <<EOF
.. _${cmd}:

${cmd}
${sep}

.. code-block:: none

${usage}
EOF

