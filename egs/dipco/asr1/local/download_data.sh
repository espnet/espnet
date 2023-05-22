#!/usr/bin/env bash

mkdir -p db

cd db  ### Note: the rest of this script is executed from the directory 'db'.

# TED-LIUM database:
if [ ! -e DiPCo ]; then
  echo "$0: downloading DIPCo data (it won't re-download if it was already downloaded.)"
  # the following command won't re-get it if it's already there
  # because of the --continue switch.
  wget --continue https://s3.amazonaws.com/dipco/DiPCo.tgz || exit 1
  tar xf "DiPCo.tgz"
else
  echo "$0: not downloading or un-tarring TEDLIUM_release2 because it already exists."
fi


exit 0
