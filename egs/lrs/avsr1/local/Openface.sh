#! /usr/bin/env bash 

# Copyright 2020 Ruhr-University (Wentao Yu)

# hand over parameters 
sdir=$1			# Directory of the dataset
savedir=$2		# Output directory where results should be saved
dset=$3			# Type of dataset (Train, Test, Val, Pretrain)
OPENFACEDIR=$4		# OpenFace build directory
corpus=$5		# LRS2 or LRS3 corpus
nj=$6
ifdebug=${7:-true}  	# if debug mode should be used, default is true

# general configuration
nj=8
if [[ "$corpus" == "LRS2" ]] ; then
    if [ "$dset" = "pretrain" ] ; then
    	sourcedir=$sdir/data/lrs2_v1/mvlrs_v1/pretrain
    else
    	sourcedir=$sdir/data/lrs2_v1/mvlrs_v1/main
    fi
    filelist=data/METADATA/Filelist_${dset}
elif [[ "$corpus" == "LRS3" ]] ; then
    sourcedir=$sdir/pretrain
    filelist=data/METADATA/Filelist_LRS3${dset}
fi

function facerecog () {
    # general configuration
    inputdir=$2/$1.mp4
    nameA="$(cut -d'/' -f1 <<<$1)"
    nameB="$(cut -d'/' -f2 <<<$1)"
    outputdir=$3/$nameA
    mkdir -p $outputdir
	
    # OpenFace feature extraction
    $4/FeatureExtraction -rigid -f $inputdir -out_dir $outputdir 

    # Cleanup output directory
    for x in $outputdir/*{.txt,.hog,.avi}
    do
        rm -rf $x           
    done
}


i=0
(
cat $filelist | while read line
do 
    ((i=i%nj)); ((i++==0)) && wait

    # debug message
    if [ "$ifdebug" = true ] ; then
        echo "Run OpenFace feature extraction for $line in $dset set"
    fi
    # run facerecog
    facerecog $line $sourcedir $savedir $OPENFACEDIR >> $savedir/log.txt &
done
)

# wait until all child processes are done
wait

echo "All background processes are done!"
find $savedir -name '*_aligned' | xargs rm -rf # comment out if you want to save features


exit 0
