#!/bin/bash
# Copyright  2017  Atlas Guide (Author : Lucas Jo)
#
# Apache 2.0
#

# Modified by Hoon Chung 2020 (ETRI) 
set -euo pipefail

echo "#### installing morfessor"
dirname=morfessor
rm -rf "${dirname}"
mkdir -p ./"${dirname}"
git clone https://github.com/aalto-speech/morfessor.git "${dirname}"
echo >&2 "installation of MORFESSOR finished successfully"
