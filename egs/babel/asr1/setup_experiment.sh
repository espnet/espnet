#!/usr/bin/env bash

if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./setup_experiment.sh <expname>"
  echo >&2 ""
  echo >&2 "Sets up an experiment to be run in the babel directory with the "
  echo >&2 "provided name."
  exit 1;
fi

expname=$1
cd ..
mkdir ${expname}
cd ${expname}

cp ../asr1/{cmd,path,run}.sh .
cp -P ../asr1/steps .
cp -P ../asr1/utils .
ln -s ../asr1/local .
ln -s ../asr1/conf .
