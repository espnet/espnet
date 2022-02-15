#! /usr/bin/env bash 

# Copyright 2020 Ruhr-University (Wentao Yu)



# hand over parameters 
sdir=$1					# source directory of the data
dset=$2					# dataset part (Train, Test, Val, pretrain)
ifsegment=$3				# if do segmentation for pretrain set
ifmulticore=${4:-true}  		# if multi cpu processing, default is true
ifdebug=$5				# with debug, we only use $num utterances from pretrain and $num Utts from Train set
num=$6					# number of utterances used from pretrain and Train set
nj=$7

# general configuration
stage=1                                 # set starting stage
stop_stage=1                            # set stop stage
sourcedir=$sdir/data/lrs2_v1/mvlrs_v1   # main data dir of LRS2 dataset, source for Video data
datadir=data/audio/clean/LRS2/$dset     # datadir of the clean audio dataset 
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
	cp $sdir/Filelist_${dset} $metadir
    fi	
    if [ "$dset" = Test ] ; then
	mv $metadir/Filelist_Test ${tmpdir}/filelists/Filelist_Test
	cat ${tmpdir}/filelists/Filelist_Test | cut -d " " -f1 > $metadir/Filelist_Test
    fi
fi

if [ "$ifdebug" = true ] ; then
    if [ "$dset" = pretrain ] ; then
	cat $metadir/Filelist_pretrain | sed -n "1,${num}p" >> $metadir/new
	rm -rf $metadir/Filelist_pretrain
	mv $metadir/new $metadir/Filelist_pretrain
    fi
    if [ "$dset" = Train ] ; then
	cat $metadir/Filelist_Train | sed -n "1,${num}p" >> $metadir/new
	rm -rf $metadir/Filelist_Train
	mv $metadir/new $metadir/Filelist_Train
   fi
   if [ "$dset" = Test ] ; then
	cat $metadir/Filelist_Test | sed -n "1,${num}p" >> $metadir/new
	rm -rf $metadir/Filelist_Test
	mv $metadir/new $metadir/Filelist_Test
   fi
    if [ "$dset" = Val ] ; then
	cat $metadir/Filelist_Val | sed -n "1,${num}p" >> $metadir/new
	rm -rf $metadir/Filelist_Val
	mv $metadir/new $metadir/Filelist_Val
   fi
fi



if [ "$dset" = pretrain ] ; then
    echo "pretrain"
    if [ "$ifsegment" = false ] ; then
        mkdir -p Dataset_processing/LRS2/pretrainsegment
        python3 -u local/preppretrainaudio.py  $sourcedir/pretrain $metadir $datadir $dset Dataset_processing/LRS2/pretrainsegment $ifmulticore $ifsegment || exit 1;
        for file in text utt2spk wav.scp; do
	    sort -u $datadir/$file -o $datadir/$file || exit 1;
        done
    else
        python3 -u local/pretrain.py  $sourcedir/pretrain $metadir $datadir $dset $nj $ifsegment || exit 1;
        for file in text utt2spk wav.scp segments; do
	    sort -u $datadir/$file -o $datadir/$file || exit 1;
        done
	python3 local/creatsegfile.py $datadir \
			   $sourcedir ${dset} $ifmulticore || exit 1;

    fi
else
   echo $dset
   python3 -u local/prepaudio.py  $sourcedir/main $metadir $datadir $dset $ifmulticore || exit 1;
   for file in text utt2spk wav.scp; do
	sort -u $datadir/$file -o $datadir/$file || exit 1;
   done
fi

exit 0
