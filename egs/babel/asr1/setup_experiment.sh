#!/bin/bash

if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./setup_experiment.sh <expname>"
  echo >&2 ""
  echo >&2 "Sets up an experiment to be run in the babel directory with the "
  echo >&2 "provided name."
  exit 1;
fi

expdir=$1

pwd=$PWD

mkdir -p ${expdir}; cd ${expdir}

cp $pwd/{cmd,path,run,run.new}.sh ./
for f in steps utils local conf; do
    ln -rs $pwd/$f ./
done
