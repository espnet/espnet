#!/usr/bin/env bash

. ./path.sh


help_message=$(cat << EOF
Usage: $0 [L3DAS22 path] [split path]
  required argument:
    L3DAS22 path: path to the L3DAS22 directory
    split path: path to split the L3DAS22 wavfiles directory
    NOTE:
      You can download L3DAS22 manually from
        https://www.kaggle.com/datasets/l3dasteam/l3das22
EOF
)



if [ $# -ne 2 ]; then
  echo "${help_message}"
  exit 1;
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# configs
nj=16
. ./cmd.sh || exit 1;

. utils/parse_options.sh

L3DAS22=$1
splitdir=$2

# check if the wav dirs exist.
for ddir in L3DAS22_Task1_dev L3DAS22_Task1_test L3DAS22_Task1_train100 L3DAS22_Task1_train360_1 L3DAS22_Task1_train360_2; do
  f=$L3DAS22/${ddir}
  if [ ! -d $f ]; then
    echo "Error: $f is not a directory."
    exit 1;
  fi
done

dev_dir=$L3DAS22/L3DAS22_Task1_dev
test_dir=$L3DAS22/L3DAS22_Task1_test
train_dir100=$L3DAS22/L3DAS22_Task1_train100
train_dir360_1=$L3DAS22/L3DAS22_Task1_train360_1
train_dir360_2=$L3DAS22/L3DAS22_Task1_train360_2


echo "Spliting multi-channel data"

trap 'rm -rf "$tmpdir"' EXIT
tmpdir=$(mktemp -d /tmp/l3das22.XXXXXX)

find $train_dir100 -name '*A.wav' -o -name '*B.wav' | sort -u | awk '{n=split($1, lst, "/"); split(lst[n], wav, "."); print(wav[1],$1)}' > $tmpdir/tr100.flist
find $train_dir360_1 -name '*A.wav' -o -name '*B.wav' | sort -u | awk '{n=split($1, lst, "/"); split(lst[n], wav, "."); print(wav[1],$1)}' > $tmpdir/tr360.flist
find $train_dir360_2 -name '*A.wav' -o -name '*B.wav' | sort -u | awk '{n=split($1, lst, "/"); split(lst[n], wav, "."); print(wav[1],$1)}' >> $tmpdir/tr360.flist
find $dev_dir -name '*A.wav' -o -name '*B.wav' | sort -u | awk '{n=split($1, lst, "/"); split(lst[n], wav, "."); print(wav[1],$1)}' > $tmpdir/dev.flist
find $test_dir -name '*A.wav' -o -name '*B.wav' | sort -u | awk '{n=split($1, lst, "/"); split(lst[n], wav, "."); print(wav[1],$1)}' > $tmpdir/test.flist


# create multi-channel files
for x in dev test tr100 tr360; do
  ddir=L3DAS22_Task1_${x}
  mkdir -p $splitdir/${ddir}

  #create sh for sox
  rm $tmpdir/${x}.sh 2>/dev/null || true
  for channel in {1..4}; do
    awk -v out="$splitdir/$ddir" -v ch="$channel" \
    '{print "sox", $2, out"/"$1"_CH"ch".wav", "remix", ch }' $tmpdir/${x}.flist >> $tmpdir/${x}.sh
  done

  # split sox files
  rm -r $tmpdir/log/${x} 2>/dev/null || true
  mkdir -p $tmpdir/log/${x}
  split -n l/$nj  -d $tmpdir/${x}.sh $tmpdir/log/${x}/
  ls $tmpdir/log/${x}/* | awk -v dir="$tmpdir/log/${x}" -v x="$x" '{n=split($1, lst, "/"); print  $1, dir"/sox_"(lst[n]+1) ".sh"}' | xargs -n2 mv
  chmod +x $tmpdir/log/${x}/*

  # sox command to separate multi-channels
  ${train_cmd} JOB=1:$nj $tmpdir/log/${x}/sox.JOB.log $tmpdir/log/${x}/sox_JOB.sh
done
