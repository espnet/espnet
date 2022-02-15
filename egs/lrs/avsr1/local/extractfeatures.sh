#! /usr/bin/env bash 

# Copyright 2020 Ruhr-University (Wentao Yu)

# hand over parameters 
sdir=$1				# source directory of video frame pictures
savedir=$2			# savedirectory for features
pretrainedmodeldir=$3		# Path to pretrained video model
dset=$4				# dataset part (Train, Test, Val, pretrain)
ifcuda=$5			# if use cuda
ifdebug=${6:-true}  	        # if debug mode should be used, default is true

mkdir -p $savedir

# running extractvfeatures script
python3 -u local/extractvfeatures.py $sdir $savedir $pretrainedmodeldir $dset $ifcuda $ifdebug  || exit 1;

exit 0
