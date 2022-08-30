nj=$1
nbest=$2
dir=$3
data=$4
lang=de
mkdir -p $dir/diversity
for n in $( seq 1 $nbest ); do
    for f in token token_int score text; do
        for i in $(seq "${nj}"); do
            cat "${dir}/logdir/output.${i}/${n}best_recog/${f}"
        done | LC_ALL=C sort -k1 >"${dir}/diversity/${f}_${n}"
    done
    paste \
        <(<"${dir}/diversity/text_${n}"  \
                python -m espnet2.bin.tokenize_text  \
                    -f 2- --input - --output - \
                    --token_type word \
                    ) \
        <(<"${data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
            >"${dir}/diversity/${n}hyp.trn.org"

    # remove utterance id
    perl -pe 's/\([^\)]+\)//g;' "${dir}/diversity/${n}hyp.trn.org" > "${dir}/diversity/${n}hyp.trn"

    # detokenizer
    sed -i 's/^[ \t]*//;s/[ \t]*$//' "${dir}/diversity/${n}hyp.trn"
    detokenizer.perl -l ${lang} -q < "${dir}/diversity/${n}hyp.trn" > "${dir}/diversity/${n}hyp.trn.detok"
done

python local/diversity.py --src $dir/diversity/ --nbest $nbest