#!/usr/bin/env bash

# general configuration
videodir=$1				# path of LRS3 dataset
dset=$2					# define set
filename=Filelist_$dset			# Name of Filelist
ifmulticore=$3				# if multi cpu processing, default is true in all scripts
ifsegpretrain=$4 			# if we segment pretrain set
sourcedir=$videodir			# set sourcedir for some scripts
modeltype=$5				# extract video features if the modeltype is AV or V
ifdebug=$6				# with debug, we only use $num utterances from pretrain and $num Utts from Train set
num=$7					# number of utterances used from pretrain and Train set


# Make folders
datadir=Dataset_processing/LRS3
audiodir=$datadir/audio
filelistdir=$audiodir/$dset
kaldidir=Dataset_processing/LRS3/kaldi

rm -rf $datadir $audiodir $filelistdir $kaldidir
mkdir -p $datadir $audiodir $filelistdir $kaldidir

# temporary folder for storing files
tmpdir=$(mktemp -d tmp-XXXXX)		
trap 'rm -rf ${tmpdir}' EXIT
mkdir  -p ${tmpdir}/filelists

# Create Filelist, sort it and cut to length for $num files
if ! [ -f "$filelistdir/$filename" ]; then
    echo "$filename does not exists. Create $filename"

    tmpfile=tmp
    prefix=$videodir/pretrain/
    suffix='.mp4'

    find $videodir -type f -name "*.mp4" > $tmpfile
    while read line; do
        line=${line#"$prefix"}
        line=${line%"$suffix"}
        echo $line >> ${tmpfile}_unsort
    done <$tmpfile
    sort ${tmpfile}_unsort > $filename
    rm $tmpfile
    rm ${tmpfile}_unsort
fi

mv Filelist_${dset} ${tmpdir}/filelists/Filelist_${dset}
cat ${tmpdir}/filelists/Filelist_${dset} | cut -d " " -f1 > Filelist_${dset}
if [ "$ifdebug" = true ] ; then
    mv Filelist_${dset} ${tmpdir}/filelists/Filelist_${dset}
    cat ${tmpdir}/filelists/Filelist_${dset} | sed -n "1,${num}p" >> ${tmpdir}/filelists/Filelist_${dset}_cut
    rm -rf ${tmpdir}/filelists/Filelist_${dset}
    mv ${tmpdir}/filelists/Filelist_${dset}_cut $filelistdir/$filename
else
    mv Filelist_${dset} $filelistdir/$filename
fi
############################## audio processing ###############################################


if [ "$ifsegpretrain" = true ] ; then
    echo "Create segmentinfo"
    python3 local/lrs3processing/audio/segmentinfo.py $sourcedir $datadir $filelistdir $dset $ifmulticore 
    sort $audiodir/pretrain_segmentinfo/pretrainlist -o  $audiodir/pretrain_segmentinfo/pretrainlist
    sort $audiodir/pretrain_segmentinfo/pretrain_text -o  $audiodir/pretrain_segmentinfo/pretrain_text
    sort $audiodir/pretrain_segmentinfo/pretrain_timeinfo -o  $audiodir/pretrain_segmentinfo/pretrain_timeinfo

fi


echo "Prepare Kaldi files"
if [ "$ifsegpretrain" = true ] ; then
    kaldipretraindir=$kaldidir/${dset}segment
    rm -rf $kaldipretraindir
    mkdir $kaldipretraindir
    cp $audiodir/${dset}_segmentinfo/* $kaldipretraindir
    wavdir=$(pwd)/$audiodir/${dset}segment
    sourcedir_segment=$audiodir/pretrainsegment
    python3 local/lrs3processing/audio/kaldi_prep_segment.py $sourcedir \
	${filelistdir}_segmentinfo \
        $kaldipretraindir \
        $dset \
        $ifmulticore || exit 1
    for file in text utt2spk wav.scp segments; do
        sort $kaldipretraindir/$file -o $kaldipretraindir/$file
    done
    echo "Lost $(wc -l $kaldipretraindir/delete_list | cut -c1) segments"
else
    kaldipretraindir=$kaldidir/$dset
    rm -rf $kaldipretraindir
    mkdir $kaldipretraindir
    wavdir=$(pwd)/$audiodir/$dset
    python3 local/lrs3processing/audio/kaldi_prep_nosegment.py $filelistdir/$filename \
        $kaldipretraindir \
        $wavdir \
        $sourcedir \
        $dset \
        $ifmulticore || exit 1
    for file in text utt2spk wav.scp; do
        sort $kaldipretraindir/$file -o $kaldipretraindir/$file|| exit 1
    done
fi
echo "Kaldi files created"

############################## audio processing ###############################################
echo "Make seginfo file for LRS3 ${dset} set"
python3 local/creatsegfile.py $kaldipretraindir \
			   $DATALRS3_DIR ${dset} $ifmulticore || exit 1;

exit 0


