#!/bin/bash
if [[ $1 && $2 ]]; then

  local=`pwd`/local

  mkdir -p data data/local data/$1 data/$2

  echo "Preparing $1 and $2 data"
  echo "copy wav.scp for $1 $2"

  echo "copy files from corpus/data to data/ for $1 $2"

  for x in $1 $2; do
    cp corpus/data/$x/* data/$x/.
  done

  echo "Preparing data OK."
else
  echo "ERROR: Preparing train test data failed !"
  echo "You must have forgotten to point to the correct train/test directories"
  echo "Usage: ./prepare_data.sh train test"
fi
