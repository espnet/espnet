#!/usr/bin/env bash

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url-base>"
  echo "e.g.: $0 downloads https://www.dropbox.com/s/rye2sd0wo718bj5/SuiSiann-0.2.1.tar"
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
    cd "${data}" || exit 1;
    wget $url
    tar xf SuiSiann-0.2.1.tar

    if $remove_archive; then
        echo "$0: removing $data/SuiSiann-0.2.1.tar file since --remove-archive option was supplied."
        rm $data/SuiSiann-0.2.1.tar
    fi

    cd "${cwd}" || exit 1;
    echo "$0: Successfully downloaded and un-tarred $data/SuiSiann-0.2.1.tar"
    touch ${data}/.complete
else
    echo "$0: Already exists. Skip download."
fi

exit 0;
