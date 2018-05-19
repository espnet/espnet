#!/bin/bash
#
# Copyright  2014  Nickolay V. Shmyrev
#            2014  Brno University of Technology (Author: Karel Vesely)
#            2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# To be run from one directory above this script.

. ./path.sh

export LC_ALL=C

set -e 
set -o pipefail
set -u

# Prepare: test, train,
for set in dev test train; do
  dir=data/$set.orig
  mkdir -p $dir

  # Merge transcripts into a single 'stm' file, do some mappings:
  # - <F0_M> -> <o,f0,male> : map dev stm labels to be coherent with train + test,
  # - <F0_F> -> <o,f0,female> : --||--
  # - (2) -> null : remove pronunciation variants in transcripts, keep in dictionary
  # - <sil> -> null : remove marked <sil>, it is modelled implicitly (in kaldi)
  # - (...) -> null : remove utterance names from end-lines of train
  # - it 's -> it's : merge words that contain apostrophe (if compound in dictionary, local/join_suffix.py)
  { # Add STM header, so sclite can prepare the '.lur' file
    echo ';;
;; LABEL "o" "Overall" "Overall results"
;; LABEL "f0" "f0" "Wideband channel"
;; LABEL "f2" "f2" "Telephone channel"
;; LABEL "male" "Male" "Male Talkers"
;; LABEL "female" "Female" "Female Talkers"
;;'
    # Process the STMs
    cat db/TEDLIUM_release2/$set/stm/*.stm | sort -k1,1 -k2,2 -k4,4n | \
      sed -e 's:<F0_M>:<o,f0,male>:' \
          -e 's:<F0_F>:<o,f0,female>:' \
          -e 's:([0-9])::g' \
          -e 's:<sil>::g' \
          -e 's:([^ ]*)$::' | \
      awk '{ $2 = "A"; print $0; }'
  } | local/join_suffix.py > data/$set.orig/stm

  # Prepare 'text' file
  # - {NOISE} -> [NOISE] : map the tags to match symbols in dictionary
  cat $dir/stm | grep -v -e 'ignore_time_segment_in_scoring' -e ';;' | \
    awk '{ printf ("%s-%07d-%07d", $1, $4*100, $5*100);
           for (i=7;i<=NF;i++) { printf(" %s", $i); }
           printf("\n");
         }' | tr '{}' '[]' | sort -k1,1 > $dir/text.orig

  cat $dir/text.orig | awk '{if (NF > 1) print $0}' | \
    local/normalize_transcript.pl '<NOISE>' | awk '{if (NF > 1) print $0}' \
    > $dir/text || exit 1

  # Prepare 'segments', 'utt2spk', 'spk2utt'
  cat $dir/text | cut -d" " -f 1 | awk -F"-" '{printf("%s %s %07.2f %07.2f\n", $0, $1, $2/100.0, $3/100.0)}' > $dir/segments
  cat $dir/segments | awk '{print $1, $2}' > $dir/utt2spk
  cat $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt

  # Prepare 'wav.scp', 'reco2file_and_channel'
  cat $dir/spk2utt | awk -v set=$set -v pwd=$PWD '{ printf("%s sph2pipe -f wav -p %s/db/TEDLIUM_release2/%s/sph/%s.sph |\n", $1, pwd, set, $1); }' > $dir/wav.scp
  cat $dir/wav.scp | awk '{ print $1, $1, "A"; }' > $dir/reco2file_and_channel

  # Create empty 'glm' file
  echo ';; empty.glm
  [FAKE]     =>  %HESITATION     / [ ] __ [ ] ;; hesitation token
  ' > data/$set.orig/glm

  # The training set seems to not have enough silence padding in the segmentations,
  # especially at the beginning of segments.  Extend the times.
  if [ $set == "train" ]; then
    mv data/$set.orig/segments data/$set.orig/segments.temp
    utils/data/extend_segment_times.py --start-padding=0.15 \
      --end-padding=0.1 <data/$set.orig/segments.temp >data/$set.orig/segments || exit 1
    rm data/$set.orig/segments.temp
  fi

  # Check that data dirs are okay!
  utils/validate_data_dir.sh --no-feats $dir || exit 1
done

