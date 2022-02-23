#! /usr/bin/env bash 

# Copyright 2020 Ruhr-University (Wentao Yu)

#. ./path.sh || exit 1;

# hand over parameters 
sdir=$1			# path to dataset
savedir=$2		# Save the segmented video data for every dataset
csvdir=$3		# The dir of csv File, which contain Face recognition information
audiodir=$4		# The dir which saves the audio Info
dset=$5			# dataset part (Train, Test, Val, pretrain)
corpus=$6		# LRS2 or LRS3 corpus
ifsegment=$7		# if do segmentation for pretrain set
ifmulticore=${8:-true}  # if multi cpus processing, default is true
if [ $# -gt 8 ];then
  noisetype=$9
else
  noisetype=None
fi

# general configuration
if [[ "$corpus" == "LRS2" ]] ; then
    if [[ "$noisetype" != "None" ]] ; then
	sourcedir=$sdir/$noisetype
	Confdir=$savedir/LRS2${dset}_$noisetype/Conf
	mkdir -p $Confdir
	AUdir=$savedir/LRS2${dset}_$noisetype/AUs
	mkdir -p $AUdir 
	datadir=$savedir/LRS2${dset}_$noisetype/Pics
	mkdir -p $datadir
	csvsavedir=$csvdir/LRS2${dset}_$noisetype
	savedset=$savedir/LRS2${dset}_$noisetype
    else
	if [ "$dset" = "pretrain" ] ; then
    	    sourcedir=$sdir/data/lrs2_v1/mvlrs_v1/pretrain
	else
    	    sourcedir=$sdir/data/lrs2_v1/mvlrs_v1/main
	fi
	Confdir=$savedir/LRS2${dset}/Conf
	mkdir -p $Confdir
	AUdir=$savedir/LRS2${dset}/AUs
	mkdir -p $AUdir 
	datadir=$savedir/LRS2${dset}/Pics
	mkdir -p $datadir
	csvsavedir=$csvdir/LRS2${dset}
	savedset=$savedir/LRS2${dset}
    fi
elif [[ "$corpus" == "LRS3" ]] ; then
    sourcedir=$sdir/pretrain
    Confdir=$savedir/LRS3${dset}/Conf
    mkdir -p $Confdir
    AUdir=$savedir/LRS3${dset}/AUs
    mkdir -p $AUdir 
    datadir=$savedir/LRS3${dset}/Pics
    mkdir -p $datadir
    csvsavedir=$csvdir/LRS3${dset}
    savedset=$savedir/LRS3${dset}
fi

# run python script for frame extraction
python3 -u local/segvideo.py $sourcedir $savedset $csvsavedir $audiodir/$dset $dset $corpus $ifsegment $ifmulticore

exit 0
