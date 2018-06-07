#!/bin/bash
#
# Copyright 2014  Gaurav Kumar.   Apache 2.0
# The input is a subset of the dataset in use. (*.sph files) 
# In addition the transcripts are needed as well. 
# This script is only called internally and should not be 
# used for any other purpose. A similar script for general usage 
# is local/fsp_data_prep.sh
# To be run from one directory above this script.

stage=0

export LC_ALL=C


if [ $# -lt 4 ]; then
   echo "Arguments should be the location of the Spanish Fisher Speech and Transcript Directories and the name of this partition
, and a list of files that belong to this partition . see ../run.sh for example."
   exit 1;
fi

subset=$3
dir=`pwd`/data/local/$subset/data
mkdir -p $dir
local=`pwd`/local
utils=`pwd`/utils
tmpdir=`pwd`/data/local/tmp
mkdir -p $tmpdir

. ./path.sh || exit 1; # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi
cd $dir

# Make directory of links to the WSJ disks such as 11-13.1.  This relies on the command
# line arguments being absolute pathnames.
rm -r links/ 2>/dev/null
mkdir links/
mkdir links/speech
mkdir links/transcripts
if [ ! -f $4 ]; then
  echo "Please specify a valid parition file. Could not find $4"
  exit 1;
fi
cat $4 | sed 's:.*/::g' | \
xargs -I % find $1/ -name %* | xargs -I % echo cp % links/

# Basic spot checks to see if we got the data that we needed
if [ ! -d links/LDC2010S01 -o ! -d links/LDC2010T04 ];
then
        echo "The speech and the data directories need to be named LDC2010S01 and LDC2010T04 respecti
vely"
        exit 1;
fi

if [ ! -d links/LDC2010S01/DISC1/data/speech -o ! -d links/LDC2010S01/DISC2/data/speech ];
then
        echo "Disc 1 and 2 directories missing or not properly organised within the speech data dir"
        echo "Typical format is LDC2010S01/DISC?/data/speech"
        exit 1;
fi

#Check the transcripts directories as well to see if they exist
if [ ! -d links/LDC2010T04/data/transcripts ];
then
        echo "Transcript directories missing or not properly organised"
        echo "Typical format is LDC2010T04/data/transcripts"
        exit 1;
fi

speech_d1=$dir/links/LDC2010S01/DISC1/data/speech
speech_d2=$dir/links/LDC2010S01/DISC2/data/speech
transcripts=$dir/links/LDC2010T04/data/transcripts                                 
                                                                                   
fcount_d1=`find ${speech_d1} -iname '*.sph' | wc -l`                                             
fcount_d2=`find ${speech_d2} -iname '*.sph' | wc -l`                                             
fcount_t=`find ${transcripts} -iname '*.tdf' | wc -l`                                            
#TODO:it seems like not all speech files have transcripts             
#Now check if we got all the files that we needed
if [ $fcount_d1 != 411 -o $fcount_d2 != 408 -o $fcount_t != 819 ];                 
then                                                                               
        echo "Incorrect number of files in the data directories"                   
        echo "DISC1 and DISC2 should contain 411 and 408 .sph files respectively"  
        echo "The transcripts should contain 819 files"                            
        exit 1;                                                                    
fi   

if [ $stage -le 0 ]; then
  #Gather all the speech files together to create a file list
  #TODO: Train and test split might be required
  (
      find $speech_d1 -iname '*.sph';
      find $speech_d2 -iname '*.sph';
  )  > $tmpdir/train_sph.flist

  #Get all the transcripts in one place
  find $transcripts -iname '*.tdf' > $tmpdir/train_transcripts.flist
fi

if [ $stage -le 1 ]; then
  $local/fsp_make_trans.pl $tmpdir
  mkdir -p $dir/train_all
  mv $tmpdir/reco2file_and_channel $dir/train_all/
fi

if [ $stage -le 2 ]; then                                                        
  sort $tmpdir/text.1 | grep -v '((' | \
  awk '{if (NF > 1){ print; }}' | \
  sed 's:<\s*[/]*\s*\s*for[ei][ei]g[nh]\s*\w*>::g' | \
  sed 's:<lname>\([^<]*\)<\/lname>:\1:g' | \
  sed 's:<lname[\/]*>::g' | \
  sed 's:<laugh>[^<]*<\/laugh>:[laughter]:g' | \
  sed 's:<\s*cough[\/]*>:[noise]:g' | \
  sed 's:<sneeze[\/]*>:[noise]:g' | \
  sed 's:<breath[\/]*>:[noise]:g' | \
  sed 's:<lipsmack[\/]*>:[noise]:g' | \
  sed 's:<background>[^<]*<\/background>:[noise]:g' | \
  sed -r 's:<[/]?background[/]?>:[noise]:g' | \
  #One more time to take care of nested stuff
  sed 's:<laugh>[^<]*<\/laugh>:[laughter]:g' | \
  sed -r 's:<[/]?laugh[/]?>:[laughter]:g' | \
  #now handle the exceptions, find a cleaner way to do this?
  sed 's:<foreign langenglish::g' | \
  sed 's:</foreign::g' | \
  sed -r 's:<[/]?foreing\s*\w*>::g' | \
  sed 's:</b::g' | \
  sed 's:<foreign langengullís>::g' | \
  sed 's:foreign>::g' | \
  sed 's:>::g' | \
  #How do you handle numbers?
  grep -v '()' | \
  #Now go after the non-printable characters
  sed -r 's:¿::g' > $tmpdir/text.2
  cp $tmpdir/text.2 $dir/train_all/text

  #Create segments file and utt2spk file
  ! cat $dir/train_all/text | perl -ane 'm:([^-]+)-([AB])-(\S+): || die "Bad line $_;"; print "$1-$2-$3 $1-$2\n"; ' > $dir/train_all/utt2spk \
  && echo "Error producing utt2spk file" && exit 1;

  cat $dir/train_all/text | perl -ane 'm:((\S+-[AB])-(\d+)-(\d+))\s: || die; $utt = $1; $reco = $2;
 $s = sprintf("%.2f", 0.01*$3); $e = sprintf("%.2f", 0.01*$4); print "$utt $reco $s $e\n"; ' >$dir/train_all/segments

  $utils/utt2spk_to_spk2utt.pl <$dir/train_all/utt2spk > $dir/train_all/spk2utt
fi

if [ $stage -le 3 ]; then
  cat $tmpdir/train_sph.flist | perl -ane 'm:/([^/]+)\.sph$: || die "bad line $_; ";  print "$1 $_"; ' > $tmpdir/sph.scp
  cat $tmpdir/sph.scp | awk -v sph2pipe=$sph2pipe '{printf("%s-A %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2); printf("%s-B %s -f wav -p -c 2 %s |\n", $1, sph2pipe, $2);}' | \
  sort -k1,1 -u  > $dir/train_all/wav.scp || exit 1;
fi

if [ $stage -le 4 ]; then
  # Build the speaker to gender map, the temporary file with the speaker in gender information is already created by fsp_make_trans.pl.
  cat $tmpdir/spk2gendertmp | sort | uniq > $dir/train_all/spk2gender
fi

echo "Fisher Spanish Data preparation succeeded."

exit 1;

