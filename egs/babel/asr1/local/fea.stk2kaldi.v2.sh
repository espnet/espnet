#!/bin/bash
# Copyright 2012-2014  Brno University of Technology (Author: Martin Karafiat, Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

data_in=data/train_lukasperturb
data=data-MultRDTv1/train_lukasperturb
stk_dir=data.stk/MultRDTv1/train_lukasperturb
nsplit=20
echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
  echo "Usage: $0 [--feakind-stk \$feakind] [--stkdir path or --stkconf \$stkcfg ] --data-dir data-\$feakind"
  echo "Instead: $0 $*
  echo "\$\# = $#"
#  echo "options:"
#  echo "  --tmp-root DIR  # root dir, where the tmpdirs get created"
  exit 1;
fi


 #copy_data_dir.sh $data_in $data
mkdir -p $data
echo "$0: Porting HTK features from $stk_dir/$subdir -> $data"

# gsub("^mnt/", "/mnt/", key); # is ugly hack when wavids contains full path: /mnt/....
find -L $stk_dir/ -name "*.fea"        | awk -v stk_dir="$stk_dir/" '{ key=$1; gsub(stk_dir, "", key); gsub("^mnt/", "/mnt/", key); gsub(/\.fea$/,"",key); print key,$0; }' | sort > $data/wav.scp


 for file in spk2utt utt2spk segments text wav.map stm glm reco2file_and_channel; do
  cp $data_in/$file $data
 done

 scp_tmp=$(head -1 $data/wav.scp | awk '{print $NF}')
 feakind_tmp=$(HList -h -z $scp_tmp | awk '/Sample Kind:/{print $NF}')
 if [ $feakind_tmp != ${feakind_tmp/_C/} ]; then
     dir=$( head -1 $data/wav.scp | awk '{print $2}' | sed 's:\(.*\)/.*:\1:')
     dir_uncompr=${dir/matylda3/scratch06\/tmp}.uncompress
     log_uncompr=${dir/matylda3/scratch06\/tmp}.uncompress/SFeaCat.log
     mkdir -p $dir_uncompr
     echo "$0: uncompressing fea to $dir_uncompr"
     if [ ! -e $log_uncompr.gz ]; then if awk '{print $2}' $data/wav.scp | $stkbin/SFeaCat -A -D -T 1 -V -l $dir_uncompr --TARGETKIND=USER -S /dev/stdin > $log_uncompr 2>&1; then gzip $log_uncompr; fi; fi
     sed -i "s: /.*/: $dir_uncompr/:" $data/wav.scp
     echo Done
 fi


# mkdir -p $dir_uncompr
# echo "$0: uncompressing fea to $dir_uncompr"  
# if [ ! -e $log_uncompr.gz ]; then if awk '{print $2}' $data/wav.scp | ~karafiat/STK/bin/SFeaCat -A -D -T 1 -V -l $dir_uncompr --TARGETKIND=USER -S /dev/stdin > $log_uncompr 2>&1; then gzip $log_uncompr; fi; fi 
# sed -i "s: /.*/: $dir_uncompr/:" $data/wav.scp

 #utils/fix_data_dir.sh $data

 echo "$0: Building $data/feats.ark"
 if [ ! -s $data/feats.ark ]; then
     if [ -e $data/segments ]; then
	 wdir=$data/split$nsplit; mkdir $wdir
	 for seg in $(scp.SplitList.pl -p $wdir $nsplit $data/segments); do
	     id=${seg##*_}
	     log=$wdir/feats.$id.log
	     awk 'r==0{SEG[$2]=1}r==1 && ($1 in SEG){print}' r=0 $seg r=1 $data/wav.scp > $wdir/wav.scp_$id
	     #awk '{print $2}' $seg | sort -u > $tmp
	     # Not working as it is gzipping even unfinished file
	     #echo "cd $PWD; . path.sh; if [ ! -e $log.gz ]; then if ( time -p copy-feats --htk-in=true scp:$wdir/wav.scp_$id ark:- | extract-feature-segments --max-overshoot=0.5 ark:- $seg ark,scp:$wdir/feats.$id.ark,$wdir/feats.$id.scp ) > $log 2>&1; then gzip $log; fi; fi" 
	     echo "cd $PWD; . path.sh; ( time -p copy-feats --htk-in=true scp:$wdir/wav.scp_$id ark:- | extract-feature-segments --max-overshoot=0.5 ark:- $seg ark,scp:$wdir/feats.$id.ark,$wdir/feats.$id.scp ) > $log 2>&1"
	 done > $data/feats.sge
	 manage_task.sh -q all.q@@blade -tc 5 -l ram_free=90G,mem_free=90G -sync yes $data/feats.sge
	 #cat $data/feats.err # check
	 cat $wdir/feats.*.scp > $data/feats.scp
	 nFeats=$(cat $data/feats.scp | wc -l)
	 nSeg=$(cat $data_in/segments | wc -l)
	 echo "Checking Status (wc $data/feats.scp; wc $data/segments): $nFeats == $nSeg" 
	 if [ $nFeats != $nSeg ]; then
	     er=$(awk -v nFeats=$nFeats -v nSeg=$nSeg 'BEGIN{err=100*(nSeg-nFeats)/nSeg; print err}')
	     echo "Missing ${er}% of files"
	     ok=$(awk -v er=$er -v trsh=2 'BEGIN{print (er<trsh)? "true" : "false"}')
	     if $ok; then
		 echo "WARNING: probably short utterances only; but check the log files" ;
	     else
		 echo "ERROR: Check the log files"
		 exit $rexit;
	     fi
	 else
	     echo OK
	 fi

	 #echo "cd $PWD; . path.sh; copy-feats --htk-in=true scp:$data/wav.scp ark:- | extract-feature-segments --max-overshoot=0.5 ark:- $data/segments ark,scp:$data/feats.ark,$data/feats.scp || exit 1" > $data/feaextr.sge
     else
	 echo "cd $PWD; . path.sh; time -p copy-feats --htk-in=true scp:$data/wav.scp ark,scp:$data/feats.ark,$data/feats.scp || exit 1" > $data/feaextr.sge
	 $train_cmd -l ram_free=150G,mem_free=150G $data/feaextr.log \
	     bash $data/feaextr.sge 
	 status=$?
	 if [ "$status" != 0 ]; then
	     echo "ERROR: $0: $data/feaextr.log has non 0 return status=$status"
	     rm $data/feats.ark $data/feats.scp
	     exit 1
	 fi
     fi
 else
     echo "$0: $data/feats.ark was already created. Skip."
 fi
 
 
