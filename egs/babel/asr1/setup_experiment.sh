#!/bin/bash

if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./setup_experiment.sh <expname>"
  echo >&2 ""
  echo >&2 "Sets up an experiment to be run in the babel directory with the "
  echo >&2 "provided name."
  exit 1;
fi

expdir=$1

mkdir -p ${expdir}; 

pwd=$PWD 
cd ${expdir}

cp $pwd/{cmd,path}.sh ./
for f in steps utils local conf run; do
    ln -rs $pwd/$f ./
done
