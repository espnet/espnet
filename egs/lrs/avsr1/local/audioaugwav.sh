#! /usr/bin/env bash 

# Copyright 2020 Ruhr-University (Wentao Yu)

##this script make the aug stream data in real wav data which we can listen
#<<'COMMENT'

. ./cmd.sh  # Needed for local or cluster-based processing
. ./path.sh # Needed for KALDI_ROOT, REC_ROOT and WAV_ROOT

# hand over parameters
srcdir=$1			# Path to augmented files (data/audio/augment)
savedir=$2			# Directory in which to save augmented files (Dataset_processing/Audioaugments)

# general configuration
savedsetdir=$savedir/$dset

wav-copy scp:$srcdir/wav.scp ark,scp:$savedsetdir/wavtemp.ark,$savedsetdir/wavtemp.scp


cat $savedsetdir/wavtemp.scp | while read line
do 
  A="$(cut -d' ' -f1 <<<$line)"
  B="$(cut -d' ' -f2 <<<$line)"
  wav-copy $B $savedsetdir/${A}.wav
done
rm -rf $savedsetdir/wavtemp.ark
rm -rf $savedsetdir/wavtemp.scp

