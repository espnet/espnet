#!/bin/bash 

# Copyright 2018 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

TMP=./local/tmp
mkdir -p ${TMP}
find $1 -name "*.lab" | xargs dirname | xargs dirname | sort | uniq > ${TMP}/dir_list.txt
cat ${TMP}/dir_list.txt | while read -r dir;do
    
    echo ${dir}

    # make lab_token
    echo -n > ${TMP}/lab_token.txt
    echo -n > ${TMP}/lab_token_num.txt
    echo -n > ${TMP}/lab_list.txt
    find ${dir}/lab/ -name "*.lab" | sort | while read -r f;do
        cat $f | awk '{if($3!="#"){for(i=3;i<=NF;i++){print $i}}}' | grep -v '^\s*$' >> ${TMP}/lab_token.txt
        cat $f | awk '{if($3!="#"){print NF-2}}' >> ${TMP}/lab_token_num.txt
        cat $f | awk -v f=$f '{if($3!="#"){print f,NR}}' >> ${TMP}/lab_list.txt
    done

    # make txt_token
    echo -n > ${TMP}/txt_token.txt
    find ${dir}/txt/ -name "*.txt" | while read -r f; do
        cat $f | awk '{if($1!=""){print $0}}' | tr A-Z a-z \
        | awk '{for(i=1;i<=NF;i++){if($i!="-"){print $i}}}' \
        | sed -e "s/\"//g" -e "s/mmm!//g" \
        | grep -v -e '^\s*$' -e '^,$'\
        | sed \
        -e "s/100bc/one\nhundredbc/g" \
        -e "s/200bc/two\nhundredbc/g" \
        -e "s/4,000/four\nthousand/g" \
        -e "s/7/seven/g" \
        -e "s/outside.stop/outside.\n\"stop/g" \
        | perl -0pe "s/wolf\n's/wolf's/m" \
        | perl -0pe "s/father\n's/father's/m" \
        | sed -e "s/://g" -e "s/(//g" -e "s/)//g" -e "s/\"//g" >> ${TMP}/txt_token.txt
        #| sed -e "s/,//g" -e "s/.//g" -e "s/!//g" -e "s/?//g" >> ${TMP}/txt_token.txt
    done

    # pick up the irregular lab_token
    cat ${TMP}/lab_token.txt | grep -n '_' | awk -F: '{print $1,$2}' > ${TMP}/convert_list.txt

# make the irregular txt_token
    cat ${TMP}/convert_list.txt | awk '{print $1}' | while read -r conv_n;do
        cat ${TMP}/txt_token.txt | awk -v cn=${conv_n} '{if(NR==cn){tmp=$1}else if(NR==cn+1){print tmp"_"$1}else{print $1}}' > ${TMP}/tmp.txt & wait
        cat ${TMP}/tmp.txt > ${TMP}/txt_token.txt & wait
    done

# for debug
    #paste ${TMP}/lab_token.txt ${TMP}/txt_token.txt > ${TMP}/${dir}.txt

    # make st&ed
    cat ${TMP}/lab_token_num.txt | awk '{sum=sum+$1;print sum,$1;}' > ${TMP}/lab_row_num.txt

# make lab-like txt sentence
    echo -n > ${TMP}/txt_sentence.txt
    cat ${TMP}/lab_row_num.txt | while read -r st_ed;do
        st=`echo ${st_ed} | awk '{print $1}'`
        ed=`echo ${st_ed} | awk '{print $2}'`
        head -n $st ${TMP}/txt_token.txt | tail -n $ed | awk -v eol=$ed '{printf("%s",$1);if(NR!=eol){printf(" ");}}END{printf("\n")}' >> ${TMP}/txt_sentence.txt
    done

# make lab-like txt file
    python local/make_new_lab.py ${TMP}/lab_list.txt ${TMP}/txt_sentence.txt ${dir}/new_lab
done
