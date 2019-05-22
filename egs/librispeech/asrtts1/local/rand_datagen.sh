#!/bin/bash

set -e

jsonout=dump/train_si84/deltafalse/data_rnd_284t.json
xvec=exp/xvector_nnet_1a/xvectors_train_si84
nlsyms=
dict=

. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: "
  echo "  $0 [options] <speech_srcdir> <text_srcdir> <destdir>"
  echo "e.g.:"
  echo " $0 --jsonout=data_rnd.json --xvec=xvectors_train data/train data/train data/train_1"
  echo "Options"
  echo "   --jsonout=<jsonout>     # name for output json"
  echo "   --xvec=<xvec>     # xvector dir"
  echo "   --nlsyms=<nlsyms>     # nlsyms file"
  echo "   --dict=<dict>     # dict file"
  exit 1;
fi


speech_src=$1
text_src=$2
tgt=$3
data_text_src=data/$(basename $text_src)
cp $data_text_src/text $tgt
cut -f2- -d " " $tgt/text | shuf > $tgt/text.only
text_num=`cat $tgt/text |wc -l`
feat_num=`cat $tgt/feats.scp |wc -l`
if [ "$text_num" != "$feat_num" ]; then
 num=$((text_num / feat_num | bc -l))
 echo "$num"
 awk '{a[NR]=$0}END{for (i=0; i<="'$num'"; i++){for(k in a){print a[k]}}}' $tgt/feats.scp |\
    shuf | head -n $text_num > $tgt/feats.scp.gen
else
  cat $tgt/feats.scp | shuf > $tgt/feats.scp.gen
fi

awk '{print $1}' $tgt/feats.scp.gen > $tgt/feats.ids 
paste $tgt/feats.ids $tgt/text.only > $tgt/text
paste $tgt/feats.ids $tgt/feats.ids > $tgt/utt2spk
cp $tgt/utt2spk $tgt/spk2utt
data2json_asrtts.sh --feat $tgt/feats.scp \
    --nlsyms $nlsyms \
    $tgt $dict > $tgt/data_rnd.json
cp $tgt/data_rnd.json $jsonout
bash local/update_json_asrtts.sh $jsonout $xvec/xvector.scp "none"
