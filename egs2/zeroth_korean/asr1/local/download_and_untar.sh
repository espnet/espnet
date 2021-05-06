#!/usr/bin/env bash

# Copyright 2018 Lucas Jo (Atlas Guide)
#           2018 Wonkyum Lee (Gridspace)
# Apache 2.0
set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

exists(){
	command -v "$1" >/dev/null 2>&1
}

if [ $# -ne "1" ]; then
	log "Usage: $0 <download_dir>"
	log "e.g.: $0 ./db"
	exit 1
fi

dir=$1
local_lm_dir=data/local/lm

AUDIOINFO='AUDIO_INFO'
AUDIOLIST='train_data_01 test_data_01'

log "Now download corpus ----------------------------------------------------"
if [ ! -f $dir/db.tar.gz ]; then
  if [ ! -d $dir ]; then
    mkdir -p $dir
  fi
  wget -O $dir/db.tar.gz http://www.openslr.org/resources/40/zeroth_korean.tar.gz
else
  log "  $dir/db.tar.gz already exist"
fi

log "Now extract corpus ----------------------------------------------------"
if [ ! -f $dir/$AUDIOINFO ]; then
  tar -zxvf $dir/db.tar.gz -C $dir
  else
    log "  corpus already extracted"
fi

if [ ! -d $local_lm_dir ]; then
    mkdir -p $local_lm_dir
fi
log "Check LMs files"
LMList="\
  zeroth.lm.fg.arpa.gz \
  zeroth.lm.tg.arpa.gz \
  zeroth.lm.tgmed.arpa.gz \
  zeroth.lm.tgsmall.arpa.gz \
  zeroth_lexicon \
  zeroth_morfessor.seg"

for file in $LMList; do
  if [ -f $local_lm_dir/$file ]; then
    log $file already exist
  else
    log "Linking "$file
    ln -s $PWD/$dir/$file $local_lm_dir/$file
  fi
done
log "all the files (lexicon, LM, segment model) are ready"
