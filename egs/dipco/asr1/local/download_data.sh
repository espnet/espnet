#!/usr/bin/env bash

mkdir -p db

cd db  ### Note: the rest of this script is executed from the directory 'db'.

# TED-LIUM database:
if [ ! -e DiPCo ]; then
  echo "$0: downloading DIPCo data (it won't re-download if it was already downloaded.)"
  # the following command won't re-get it if it's already there
  # because of the --continue switch.
  git clone https://huggingface.co/datasets/huckiyang/DiPCo
  # Remove .git to reduce data space.
  rm -rf DiPCo/.git
else
  echo "$0: not downloading or un-tarring DIPCo because it already exists."
fi


exit 0
