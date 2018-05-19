#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script dumps examples MPE or MMI or state-level minimum bayes risk (sMBR)
# training of neural nets.  Note: for "criterion", smbr > mpe > mmi in terms of
# compatibility of the dumped egs, meaning you can use the egs dumped with
# --criterion smbr for MPE or MMI, and egs dumped with --criterion mpe for MMI
# training.  The discriminative training program itself doesn't enforce this and
# it would let you mix and match them arbitrarily; we area speaking in terms of
# the correctness of the algorithm that splits the lattices into pieces.

# Begin configuration section.
cmd=run.pl
criterion=smbr
drop_frames=false #  option relevant for MMI, affects how we dump examples.
samples_per_iter=400000 # measured in frames, not in "examples"
max_temp_archives=128 # maximum number of temp archives per input job, only
                      # affects the process of generating archives, not the
                      # final result.

stage=0
iter=final
cleanup=true
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 6 ]; then
  echo "Usage: $0 [opts] <data> <lang> <ali-dir> <denlat-dir> <src-online-nnet2-dir> <degs-dir>"
  echo " e.g.: $0 data/train data/lang exp/nnet2_online/nnet_a_online{_ali,_denlats,_degs}"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs (probably would be good to add --max-jobs-run 5 or so if using"
  echo "                                                   # GridEngine (to avoid excessive NFS traffic)."
  echo "  --samples-per-iter <#samples|400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --stage <stage|-8>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --criterion <criterion|smbr>                     # Training criterion: may be smbr, mmi or mpfe"
  echo "  --online-ivector-dir <dir|"">                    # Directory for online-estimated iVectors, used in the"
  echo "                                                   # online-neural-net setup.  (but you may want to use"
  echo "                                                   # steps/online/nnet2/get_egs_discriminative2.sh instead)"
  exit 1;
fi

data=$1
lang=$2
alidir=$3
denlatdir=$4
srcdir=$5
dir=$6


# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/num_jobs $alidir/tree \
         $denlatdir/lat.1.gz $denlatdir/num_jobs $srcdir/$iter.mdl $srcdir/conf/online_nnet2_decoding.conf; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log $dir/info || exit 1;

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

nj=$(cat $denlatdir/num_jobs) || exit 1; # $nj is the number of
                                         # splits of the denlats and alignments.


nj_ali=$(cat $alidir/num_jobs) || exit 1;

sdata=$data/split$nj
utils/split_data.sh $data $nj




if [ $nj_ali -eq $nj ]; then
  ali_rspecifier="ark,s,cs:gunzip -c $alidir/ali.JOB.gz |"
else
  ali_rspecifier="scp:$dir/ali.scp"
  if [ $stage -le 1 ]; then
    echo "$0: number of jobs in den-lats versus alignments differ: dumping them as single archive and index."
    alis=$(for n in $(seq $nj_ali); do echo -n "$alidir/ali.$n.gz "; done)
    copy-int-vector --print-args=false \
      "ark:gunzip -c $alis|" ark,scp:$dir/ali.ark,$dir/ali.scp || exit 1;
  fi
fi


silphonelist=`cat $lang/phones/silence.csl` || exit 1;

cp $alidir/tree $dir
cp $lang/phones/silence.csl $dir/info || exit 1;
cp $srcdir/$iter.mdl $dir/final.mdl || exit 1;

grep -v '^--endpoint' $srcdir/conf/online_nnet2_decoding.conf >$dir/feature.conf || exit 1;

ivector_dim=$(online2-wav-dump-features --config=$dir/feature.conf --print-ivector-dim=true) || exit 1;

echo $ivector_dim > $dir/info/ivector_dim

! [ $ivector_dim -ge 0 ] && echo "$0: error getting iVector dim" && exit 1;

if [ -f $data/segments ]; then
  # note: in the feature extraction, because the program online2-wav-dump-features is sensitive to the
  # previous utterances within a speaker, we do the filtering after extracting the features.
  echo "$0 [info]: segments file exists: using that."
  feats="ark,s,cs:extract-segments scp:$sdata/JOB/wav.scp $sdata/JOB/segments ark:- | online2-wav-dump-features --config=$dir/feature.conf ark:$sdata/JOB/spk2utt ark,s,cs:- ark:- |"
else
  echo "$0 [info]: no segments file exists, using wav.scp."
  feats="ark,s,cs:online2-wav-dump-features --config=$dir/feature.conf ark:$sdata/JOB/spk2utt scp:$sdata/JOB/wav.scp ark:- |"
fi


if [ $stage -le 2 ]; then
  echo "$0: working out number of frames of training data"
  num_frames=$(steps/nnet2/get_num_frames.sh $data)

  echo $num_frames > $dir/info/num_frames

  # Working out total number of archives. Add one on the assumption the
  # num-frames won't divide exactly, and we want to round up.
  num_archives=$[$num_frames/$samples_per_iter + 1]

  # the next few lines relate to how we may temporarily split each input job
  # into fewer than $num_archives pieces, to avoid using an excessive
  # number of filehandles.
  archive_ratio=$[$num_archives/$max_temp_archives+1]
  num_archives_temp=$[$num_archives/$archive_ratio]
  # change $num_archives slightly to make it an exact multiple
  # of $archive_ratio.
  num_archives=$[$num_archives_temp*$archive_ratio]

  echo $num_archives >$dir/info/num_archives || exit 1
  echo $num_archives_temp >$dir/info/num_archives_temp || exit 1

  frames_per_archive=$[$num_frames/$num_archives]

  # note, this is the number of frames per archive prior to discarding frames.
  echo $frames_per_archive > $dir/info/frames_per_archive
else
  num_archives=$(cat $dir/info/num_archives) || exit 1;
  num_archives_temp=$(cat $dir/info/num_archives_temp) || exit 1;
  frames_per_archive=$(cat $dir/info/frames_per_archive) || exit 1;
fi

echo "$0: Splitting the data up into $num_archives archives (using $num_archives_temp temporary pieces per input job)"
echo "$0: giving samples-per-iteration of $frames_per_archive (you requested $samples_per_iter)."

# we create these data links regardless of the stage, as there are situations
# where we would want to recreate a data link that had previously been deleted.

if [ -d $dir/storage ]; then
  echo "$0: creating data links for distributed storage of degs"
  # See utils/create_split_dir.pl for how this 'storage' directory is created.
  for x in $(seq $nj); do
    for y in $(seq $num_archives_temp); do
      utils/create_data_link.pl $dir/degs_orig.$x.$y.ark
    done
  done
  for z in $(seq $num_archives); do
    utils/create_data_link.pl $dir/degs.$z.ark
  done
  if [ $num_archives_temp -ne $num_archives ]; then
    for z in $(seq $num_archives); do
      utils/create_data_link.pl $dir/degs_temp.$z.ark
    done
  fi
fi

if [ $stage -le 3 ]; then
  echo "$0: getting initial training examples by splitting lattices"

  degs_list=$(for n in $(seq $num_archives_temp); do echo -n "ark:$dir/degs_orig.JOB.$n.ark "; done)

  $cmd JOB=1:$nj $dir/log/get_egs.JOB.log \
    nnet-get-egs-discriminative --criterion=$criterion --drop-frames=$drop_frames \
      "$srcdir/$iter.mdl" "$feats" "$ali_rspecifier" "ark,s,cs:gunzip -c $denlatdir/lat.JOB.gz|" ark:- \| \
    nnet-copy-egs-discriminative $const_dim_opt ark:- $degs_list || exit 1;
  sleep 5;  # wait a bit so NFS has time to write files.
fi

if [ $stage -le 4 ]; then

  degs_list=$(for n in $(seq $nj); do echo -n "$dir/degs_orig.$n.JOB.ark "; done)

  if [ $num_archives -eq $num_archives_temp ]; then
    echo "$0: combining data into final archives and shuffling it"

    $cmd JOB=1:$num_archives $dir/log/shuffle.JOB.log \
      cat $degs_list \| nnet-shuffle-egs-discriminative --srand=JOB ark:- \
       ark:$dir/degs.JOB.ark || exit 1;
  else
    echo "$0: combining and re-splitting data into un-shuffled versions of final archives."

    archive_ratio=$[$num_archives/$num_archives_temp]
    ! [ $archive_ratio -gt 1 ] && echo "$0: Bad archive_ratio $archive_ratio" && exit 1;

    # note: the \$[ .. ] won't be evaluated until the job gets executed.  The
    # aim is to write to the archives with the final numbering, 1
    # ... num_archives, which is more than num_archives_temp.  The list with
    # \$[... ] expressions in it computes the set of final indexes for each
    # temporary index.
    degs_list_out=$(for n in $(seq $archive_ratio); do echo -n "ark:$dir/degs_temp.\$[((JOB-1)*$archive_ratio)+$n].ark "; done)
    # e.g. if dir=foo and archive_ratio=2, we'd have
    # degs_list_out='foo/degs_temp.$[((JOB-1)*2)+1].ark foo/degs_temp.$[((JOB-1)*2)+2].ark'

    $cmd JOB=1:$num_archives_temp $dir/log/resplit.JOB.log \
      cat $degs_list \| nnet-copy-egs-discriminative --srand=JOB ark:- \
      $degs_list_out || exit 1;
  fi
fi

if [ $stage -le 5 ] && [ $num_archives -ne $num_archives_temp ]; then
  echo "$0: shuffling final archives."

  $cmd JOB=1:$num_archives $dir/log/shuffle.JOB.log \
    nnet-shuffle-egs-discriminative --srand=JOB ark:$dir/degs_temp.JOB.ark \
      ark:$dir/degs.JOB.ark || exit 1

fi

if $cleanup; then
  echo "$0: removing temporary archives."
  for x in $(seq $nj); do
    for y in $(seq $num_archives_temp); do
      file=$dir/degs_orig.$x.$y.ark
      [ -L $file ] && rm $(utils/make_absolute.sh $file); rm $file
    done
  done
  if [ $num_archives_temp -ne $num_archives ]; then
    for z in $(seq $num_archives); do
      file=$dir/degs_temp.$z.ark
      [ -L $file ] && rm $(utils/make_absolute.sh $file); rm $file
    done
  fi
fi

echo "$0: Done."
