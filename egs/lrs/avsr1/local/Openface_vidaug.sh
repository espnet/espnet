#! /usr/bin/env bash 

# Copyright 2020 Ruhr-University (Wentao Yu)

nj=8

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

# hand over parameters 
sdir=$1			# Directory of the dataset
savedir=$2		# Output directory where results should be saved
dset=$3			# Type of dataset (Train, Test, Val, Pretrain)
OPENFACEDIR=$4		# OpenFace build directory
corpus=$5		# LRS2 or LRS3 corpus
noisetype=$6		# video noise type blur or saltandpepper
nj=$7
ifdebug=${8:-true}  	# if debug mode should be used, default is true

# general configuration


sourcedir=$sdir/$noisetype
filelist=data/METADATA/Filelist_${dset}

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
rm `find $savedir -name '*_aligned'` -rf # comment out if you want to save features


exit 0
