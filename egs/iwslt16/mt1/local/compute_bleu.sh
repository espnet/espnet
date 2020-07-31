#! /bin/sh

# Copyright 2019 Shun Kiyono
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


decode_dir=$1
src_lang=$2
tgt_lang=$3
ttask=$4
feat_trans_dir=$5

dataset_name=`echo $ttask | cut -d'.' -f1`


# concatenate splitted json
python3 local/extract_recog_text.py --path $decode_dir | sort -k1 -n | cut -f2 | sed -r 's/(@@ )|(@@ ?$)//g' > $decode_dir/hypothesis.txt

# compute bleu score
ref=$feat_trans_dir/${dataset_name}.tkn.tc.${tgt_lang}
multi-bleu.perl $ref < $decode_dir/hypothesis.txt > $decode_dir/score.txt
