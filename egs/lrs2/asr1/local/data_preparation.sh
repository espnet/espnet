#! /usr/bin/env bash

# Copyright 2020 Ruhr-University (Wentao Yu)

# hand over parameters
sdir=$1					# source directory of the data
dset=$2					# dataset part (Train, Test, Val, pretrain)
segment=$3				# if do segmentation for pretrain set
nj=$4  		                       	# if multi cpu processing, default is true

# general configuration
stage=0                                 # set starting stage
stop_stage=100                          # set stop stage
sourcedir=$sdir/data/lrs2_v1/mvlrs_v1   # main data dir of LRS2 dataset, source for Video data
datadir=data/$dset     			# datadir of the clean audio dataset
metadir=data/METADATA			# datadir of the metadata

mkdir -p $datadir

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # copy the Dataset metadata
    tmpdir=$(mktemp -d tmp-XXXXX)
    trap 'rm -rf ${tmpdir}' EXIT
    mkdir -p ${tmpdir}/filelists
    mkdir -p $metadir
    if [ "$dset" = Val ] ; then
	if [ ! -f "$metadir/Filelist_Val" ]; then
	    cp $sdir/Filelist_${dset} $metadir
	fi
    else
	cp -f $sdir/Filelist_${dset} $metadir
    fi
    if [ "$dset" = Test ] ; then
	mv $metadir/Filelist_Test ${tmpdir}/filelists/Filelist_Test
	cat ${tmpdir}/filelists/Filelist_Test | cut -d " " -f1 > $metadir/Filelist_Test
    fi
fi

if [ "$dset" = pretrain ] ; then
    echo "pretrain"
    segmentdir=data/Dataset_processing/pretrainsegment
    mkdir -p $segmentdir
    python3 -u local/pretrain.py  $sourcedir/pretrain $metadir $datadir $dset $nj $segment || exit 1;
else
   echo $dset
   ### Generate the text, utt2spk and wav.scp file
   python3 -u local/make_files.py  $sourcedir/main $metadir $datadir $dset $nj || exit 1;
   for file in text utt2spk wav.scp; do
	sort -u $datadir/$file -o $datadir/$file || exit 1;
   done
fi

exit 0
