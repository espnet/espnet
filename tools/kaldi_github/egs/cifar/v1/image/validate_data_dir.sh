#!/bin/bash


no_feats=false
no_image=false
no_text=false
no_spk_sort=false

for x in `seq 4`; do
  if [ "$1" == "--no-feats" ]; then
    no_feats=true
    shift;
  fi
  if [ "$1" == "--no-text" ]; then
    no_text=true
    shift;
  fi
  if [ "$1" == "--no-image" ]; then
    no_image=true
    shift;
  fi
  if [ "$1" == "--no-spk-sort" ]; then
    no_spk_sort=true
    shift;
  fi
done

if [ $# -ne 1 ]; then
  echo "Usage: $0 [--no-feats] [--no-text] [--no-image] [--no-spk-sort] <data-dir>"
  echo "The --no-xxx options mean that the script does not require "
  echo "xxx.scp to be present, but it will check it if it is present."
  echo "--no-spk-sort means that the script does not require the utt2spk to be "
  echo "sorted by the speaker-id in addition to being sorted by utterance-id."
  echo "By default, utt2spk is expected to be sorted by both, which can be "
  echo "achieved by making the speaker-id prefixes of the utterance-ids"
  echo "e.g.: $0 data/train"
  exit 1;
fi

data=$1

if [ ! -d $data ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

for f in spk2utt utt2spk; do
  if [ ! -f $data/$f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
  if [ ! -s $data/$f ]; then
    echo "$0: empty file $f"
    exit 1;
  fi
done

! cat $data/utt2spk | awk '{if (NF != 2) exit(1); }' && \
  echo "$0: $data/utt2spk has wrong format." && exit;

ns=$(wc -l < $data/spk2utt)
if [ "$ns" == 1 ]; then
  echo "$0: WARNING: you have only one speaker.  This probably a bad idea."
  echo "   Search for the word 'bold' in http://kaldi-asr.org/doc/data_prep.html"
  echo "   for more information."
fi


tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT HUP INT PIPE TERM

export LC_ALL=C

function check_sorted_and_uniq {
  ! awk '{print $1}' $1 | sort | uniq | cmp -s - <(awk '{print $1}' $1) && \
    echo "$0: file $1 is not in sorted order or has duplicates" && exit 1;
}

function partial_diff {
  diff $1 $2 | head -n 6
  echo "..."
  diff $1 $2 | tail -n 6
  n1=`cat $1 | wc -l`
  n2=`cat $2 | wc -l`
  echo "[Lengths are $1=$n1 versus $2=$n2]"
}

check_sorted_and_uniq $data/utt2spk

if ! $no_spk_sort; then
  ! cat $data/utt2spk | sort -k2 | cmp -s - $data/utt2spk && \
     echo "$0: utt2spk is not in sorted order when sorted first on speaker-id " && \
     echo "(fix this by making speaker-ids prefixes of utt-ids)" && exit 1;
fi

check_sorted_and_uniq $data/spk2utt

! cmp -s <(cat $data/utt2spk | awk '{print $1, $2;}') \
     <(utils/spk2utt_to_utt2spk.pl $data/spk2utt)  && \
   echo "$0: spk2utt and utt2spk do not seem to match" && exit 1;

cat $data/utt2spk | awk '{print $1;}' > $tmpdir/utts

if [ ! -f $data/text ] && ! $no_text; then
  echo "$0: no such file $data/text (if this is by design, specify --no-text)"
  exit 1;
fi

num_utts=`cat $tmpdir/utts | wc -l`
if [ -f $data/text ]; then
  utils/validate_text.pl $data/text || exit 1;
  check_sorted_and_uniq $data/text
  text_len=`cat $data/text | wc -l`
  illegal_sym_list="<s> </s> #0"
  for x in $illegal_sym_list; do
    if grep -w "$x" $data/text > /dev/null; then
      echo "$0: Error: in $data, text contains illegal symbol $x"
      exit 1;
    fi
  done
  awk '{print $1}' < $data/text > $tmpdir/utts.txt
  if ! cmp -s $tmpdir/utts{,.txt}; then
    echo "$0: Error: in $data, utterance lists extracted from utt2spk and text"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/utts{,.txt}
    exit 1;
  fi
fi

if [ -f $data/segments ] && [ ! -f $data/images.scp ]; then
  echo "$0: in directory $data, segments file exists but no images.scp"
  exit 1;
fi


if [ ! -f $data/images.scp ] && ! $no_image; then
  echo "$0: no such file $data/images.scp (if this is by design, specify --no-image)"
  exit 1;
fi

if [ -f $data/images.scp ]; then
  check_sorted_and_uniq $data/images.scp

  if grep -E -q '^\S+\s+~' $data/images.scp; then
    # note: it's not a good idea to have any kind of tilde in images.scp, even if
    # part of a command, as it would cause compatibility problems if run by
    # other users, but this used to be not checked for so we let it slide unless
    # it's something of the form "foo ~/foo.wav" (i.e. a plain file name) which
    # would definitely cause problems as the fopen system call does not do
    # tilde expansion.
    echo "$0: Please do not use tilde (~) in your images.scp."
    exit 1;
  fi

  if [ -f $data/segments ]; then

    check_sorted_and_uniq $data/segments
    # We have a segments file -> interpret wav file as "recording-ids" not utterance-ids.
    ! cat $data/segments | \
      awk '{if (NF != 4 || $4 <= $3) { print "Bad line in segments file", $0; exit(1); }}' && \
      echo "$0: badly formatted segments file" && exit 1;

    segments_len=`cat $data/segments | wc -l`
    if [ -f $data/text ]; then
      ! cmp -s $tmpdir/utts <(awk '{print $1}' <$data/segments) && \
        echo "$0: Utterance list differs between $data/utt2spk and $data/segments " && \
        echo "$0: Lengths are $segments_len vs $num_utts" && \
        exit 1
    fi

    cat $data/segments | awk '{print $2}' | sort | uniq > $tmpdir/recordings
    awk '{print $1}' $data/images.scp > $tmpdir/recordings.wav
    if ! cmp -s $tmpdir/recordings{,.wav}; then
      echo "$0: Error: in $data, recording-ids extracted from segments and images.scp"
      echo "$0: differ, partial diff is:"
      partial_diff $tmpdir/recordings{,.wav}
      exit 1;
    fi
    if [ -f $data/reco2file_and_channel ]; then
      # this file is needed only for ctm scoring; it's indexed by recording-id.
      check_sorted_and_uniq $data/reco2file_and_channel
      ! cat $data/reco2file_and_channel | \
        awk '{if (NF != 3 || ($3 != "A" && $3 != "B" )) {
                if ( NF == 3 && $3 == "1" ) {
                  warning_issued = 1;
                } else {
                  print "Bad line ", $0; exit 1;
                }
              }
            }
            END {
              if (warning_issued == 1) {
                print "The channel should be marked as A or B, not 1! You should change it ASAP! "
              }
            }' && echo "$0: badly formatted reco2file_and_channel file" && exit 1;
      cat $data/reco2file_and_channel | awk '{print $1}' > $tmpdir/recordings.r2fc
      if ! cmp -s $tmpdir/recordings{,.r2fc}; then
        echo "$0: Error: in $data, recording-ids extracted from segments and reco2file_and_channel"
        echo "$0: differ, partial diff is:"
        partial_diff $tmpdir/recordings{,.r2fc}
        exit 1;
      fi
    fi
  else
    # No segments file -> assume images.scp indexed by utterance.
    cat $data/images.scp | awk '{print $1}' > $tmpdir/utts.wav
    if ! cmp -s $tmpdir/utts{,.wav}; then
      echo "$0: Error: in $data, utterance lists extracted from utt2spk and images.scp"
      echo "$0: differ, partial diff is:"
      partial_diff $tmpdir/utts{,.wav}
      exit 1;
    fi

    if [ -f $data/reco2file_and_channel ]; then
      # this file is needed only for ctm scoring; it's indexed by recording-id.
      check_sorted_and_uniq $data/reco2file_and_channel
      ! cat $data/reco2file_and_channel | \
        awk '{if (NF != 3 || ($3 != "A" && $3 != "B" )) {
                if ( NF == 3 && $3 == "1" ) {
                  warning_issued = 1;
                } else {
                  print "Bad line ", $0; exit 1;
                }
              }
            }
            END {
              if (warning_issued == 1) {
                print "The channel should be marked as A or B, not 1! You should change it ASAP! "
              }
            }' && echo "$0: badly formatted reco2file_and_channel file" && exit 1;
      cat $data/reco2file_and_channel | awk '{print $1}' > $tmpdir/utts.r2fc
      if ! cmp -s $tmpdir/utts{,.r2fc}; then
        echo "$0: Error: in $data, utterance-ids extracted from segments and reco2file_and_channel"
        echo "$0: differ, partial diff is:"
        partial_diff $tmpdir/utts{,.r2fc}
        exit 1;
      fi
    fi
  fi
fi

if [ ! -f $data/feats.scp ] && ! $no_feats; then
  echo "$0: no such file $data/feats.scp (if this is by design, specify --no-feats)"
  exit 1;
fi

if [ -f $data/feats.scp ]; then
  check_sorted_and_uniq $data/feats.scp
  cat $data/feats.scp | awk '{print $1}' > $tmpdir/utts.feats
  if ! cmp -s $tmpdir/utts{,.feats}; then
    echo "$0: Error: in $data, utterance-ids extracted from utt2spk and features"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/utts{,.feats}
    exit 1;
  fi
fi


if [ -f $data/cmvn.scp ]; then
  check_sorted_and_uniq $data/cmvn.scp
  cat $data/cmvn.scp | awk '{print $1}' > $tmpdir/speakers.cmvn
  cat $data/spk2utt | awk '{print $1}' > $tmpdir/speakers
  if ! cmp -s $tmpdir/speakers{,.cmvn}; then
    echo "$0: Error: in $data, speaker lists extracted from spk2utt and cmvn"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/speakers{,.cmvn}
    exit 1;
  fi
fi

if [ -f $data/spk2gender ]; then
  check_sorted_and_uniq $data/spk2gender
  ! cat $data/spk2gender | awk '{if (!((NF == 2 && ($2 == "m" || $2 == "f")))) exit 1; }' && \
     echo "$0: Mal-formed spk2gender file" && exit 1;
  cat $data/spk2gender | awk '{print $1}' > $tmpdir/speakers.spk2gender
  cat $data/spk2utt | awk '{print $1}' > $tmpdir/speakers
  if ! cmp -s $tmpdir/speakers{,.spk2gender}; then
    echo "$0: Error: in $data, speaker lists extracted from spk2utt and spk2gender"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/speakers{,.spk2gender}
    exit 1;
  fi
fi

if [ -f $data/spk2warp ]; then
  check_sorted_and_uniq $data/spk2warp
  ! cat $data/spk2warp | awk '{if (!((NF == 2 && ($2 > 0.5 && $2 < 1.5)))){ print; exit 1; }}' && \
     echo "$0: Mal-formed spk2warp file" && exit 1;
  cat $data/spk2warp | awk '{print $1}' > $tmpdir/speakers.spk2warp
  cat $data/spk2utt | awk '{print $1}' > $tmpdir/speakers
  if ! cmp -s $tmpdir/speakers{,.spk2warp}; then
    echo "$0: Error: in $data, speaker lists extracted from spk2utt and spk2warp"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/speakers{,.spk2warp}
    exit 1;
  fi
fi

if [ -f $data/utt2warp ]; then
  check_sorted_and_uniq $data/utt2warp
  ! cat $data/utt2warp | awk '{if (!((NF == 2 && ($2 > 0.5 && $2 < 1.5)))){ print; exit 1; }}' && \
     echo "$0: Mal-formed utt2warp file" && exit 1;
  cat $data/utt2warp | awk '{print $1}' > $tmpdir/utts.utt2warp
  cat $data/utt2spk | awk '{print $1}' > $tmpdir/utts
  if ! cmp -s $tmpdir/utts{,.utt2warp}; then
    echo "$0: Error: in $data, utterance lists extracted from utt2spk and utt2warp"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/utts{,.utt2warp}
    exit 1;
  fi
fi

# check some optionally-required things
for f in vad.scp utt2lang utt2uniq; do
  if [ -f $data/$f ]; then
    check_sorted_and_uniq $data/$f
    if ! cmp -s <( awk '{print $1}' $data/utt2spk ) \
      <( awk '{print $1}' $data/$f ); then
      echo "$0: error: in $data, $f and utt2spk do not have identical utterance-id list"
      exit 1;
    fi
  fi
done


if [ -f $data/utt2dur ]; then
  check_sorted_and_uniq $data/utt2dur
  cat $data/utt2dur | awk '{print $1}' > $tmpdir/utts.utt2dur
  if ! cmp -s $tmpdir/utts{,.utt2dur}; then
    echo "$0: Error: in $data, utterance-ids extracted from utt2spk and utt2dur file"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/utts{,.utt2dur}
    exit 1;
  fi
  cat $data/utt2dur | \
    awk '{ if (NF != 2 || !($2 > 0)) { print "Bad line : " $0; exit(1) }}' || exit 1
fi


echo "$0: Successfully validated data-directory $data"
