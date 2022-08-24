#!/usr/bin/env bash

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url-base>"
  echo "e.g.: $0 /export/a05/xna/data www.openslr.org/resources/38"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
fi

data=$1
url=$2

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

if [ -z "$url" ]; then
  echo "$0: empty URL base."
  exit 1;
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e "${data}/.complete" ]; then
    mkdir -p "${data}"
    cd "${data}" || exit 1;
    wget $url/ST-CMDS-20170001_1-OS.tar.gz
    tar xf ST-CMDS-20170001_1-OS.tar.gz

    if $remove_archive; then
        echo "$0: removing $data/ST-CMDS-20170001_1-OS.tar.gz file since --remove-archive option was supplied."
        rm $data/ST-CMDS-20170001_1-OS.tar.gz
    fi

    cd "${cwd}" || exit 1;
    echo "$0: Successfully downloaded and un-tarred $data/ST-CMDS-20170001_1-OS.tar.gz"
    touch ${data}/.complete
else
    echo "$0: Already exists. Skip download."
fi

exit 0;
