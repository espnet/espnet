#!/bin/bash
# Copyright  2017  Atlas Guide (Author : Lucas Jo)
#
# Apache 2.0
#

# Modified by Hoon Chung 2020 (ETRI) 

echo "#### installing morfessor"
dirname=morfessor
if [ ! -d ./$dirname ]; then
  mkdir -p ./$dirname
  git clone https://github.com/aalto-speech/morfessor.git morfessor ||
    {
      echo  >&2 "$0: Error git clone operation "
      echo  >&2 "  Failed in cloning the github repository (https://github.com/aalto-speech/morfessor.git)"
      exit
    }
fi

echo >&2 "installation of MORFESSOR finished successfully"
