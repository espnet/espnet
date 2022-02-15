#! /usr/bin/env bash 

# Copyright 2020 Ruhr-University (Wentao Yu)

# hand over parameters 
savematdir=$1		# savematdir
saveptdir=$2		# saveptdir
srcdir=$3		# path to the audio mp3 files (augmented audio files)
dset=$4			# dataset part (Train, Test, Val, pretrain)
ifmulticore=${7:-true}	# if multi cpus processing, default is true


if [ "$dset" = pretrain ] || [ "$dset" = Train ] ; then
	mkdir -p $savematdir/${dset}_filelists
	mkdir -p $savematdir/${dset}
	ls $srcdir/$dset > $savematdir/${dset}_filelists/${dset}.txt
	split -l 5000 $savematdir/${dset}_filelists/${dset}.txt $savematdir/${dset}_filelists/${dset}
	rm -rf $savematdir/${dset}_filelists/${dset}.txt
	mkdir -p DeepXi/set/test_noisy_speech || exit 1;
	for filename in $savematdir/${dset}_filelists/${dset}*; do
	    cat $filename | while read line
	    do 	
  	        mv $srcdir/$dset/$line DeepXi/set/test_noisy_speech || exit 1;
	    done
	    cd DeepXi
	    ./run.sh VER="mhanet-1.1c" INFER=1 GAIN="mmse-lsa" OUT_TYPE="xi_hat" || exit 1;
	    cd ..
	    mv DeepXi/set/test_noisy_speech/* $srcdir/$dset
	    mv DeepXi/out/mhanet-1.1c/e200/xi_hat/* $savematdir/$dset

	    python3 -u local/convertsnr.py $savematdir/$dset $saveptdir/$dset $ifmulticore || exit 1;
	    rm -rf $savematdir/$dset/*
        done
	rm -rf $savematdir/${dset}_filelists

else

	mv $srcdir/$dset DeepXi/set/test_noisy_speech
	cd DeepXi
	./run.sh VER="mhanet-1.1c" INFER=1 GAIN="mmse-lsa" OUT_TYPE="xi_hat" || exit 1;
	cd ..
	mv DeepXi/set/test_noisy_speech $srcdir/$dset
	mv DeepXi/out/mhanet-1.1c/e200/xi_hat/* $savematdir/$dset

	python3 -u local/convertsnr.py $savematdir/$dset $saveptdir/$dset $ifmulticore || exit 1;
fi
