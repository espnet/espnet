#!/usr/bin/env bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <src-dir> <src-lang> <tgt_text-lang>"
    echo "e.g.: $0 /n//work3/inaguma/mTEDx source_lang target_lang"
    exit 1;
fi

src_lang=$2
tgt_lang=$3

# all utterances are FLAC compressed
# if ! which flac >&/dev/null; then
#    echo "Please install 'flac' on ALL worker nodes!"
#    exit 1
# fi

for set in train valid test; do
    src=$1/${src_lang}-${tgt_lang}/data/${set}
    dst=data/local/${src_lang}-${tgt_lang}/${set}

    [ ! -d ${src} ] && echo "$0: no such directory ${src}" && exit 1;

    wav_dir=${src}/wav
    trans_dir=${src}/txt
    yml=${trans_dir}/${set}.yaml
    src_text=${trans_dir}/${set}.${src_lang}
    tgt_text=${trans_dir}/${set}.${tgt_lang}

    mkdir -p ${dst} || exit 1;

    [ ! -d ${wav_dir} ] && echo "$0: no such directory ${wav_dir}" && exit 1;
    [ ! -d ${trans_dir} ] && echo "$0: no such directory ${trans_dir}" && exit 1;
    [ ! -f ${yml} ] && echo "$0: expected file ${yml} to exist" && exit 1;
    [ ! -f ${src_text} ] && echo "$0: expected file ${src_text} to exist" && exit 1;
    [ ! -f ${tgt_text} ] && echo "$0: expected file ${tgt_text} to exist" && exit 1;

    wav_scp=${dst}/wav.scp; [[ -f "${wav_scp}" ]] && rm ${wav_scp}
    trans_src=${dst}/text.${src_lang}; [[ -f "${trans_src}" ]] && rm ${trans_src}
    trans_tgt=${dst}/text.${tgt_lang}; [[ -f "${trans_tgt}" ]] && rm ${trans_tgt}
    utt2spk=${dst}/utt2spk; [[ -f "${utt2spk}" ]] && rm ${utt2spk}
    spk2utt=${dst}/spk2utt; [[ -f "${spk2utt}" ]] && rm ${spk2utt}
    segments=${dst}/segments; [[ -f "${segments}" ]] && rm ${segments}

    # error check
    n=$(cat ${yml} | grep duration | wc -l)
    n_src=$(cat ${src_text} | wc -l)
    n_tgt=$(cat ${tgt_text} | wc -l)
    [ ${n} -ne ${n_src} ] && echo "Warning: expected ${n} data files, found ${n_src}" && exit 1;
    [ ${n} -ne ${n_tgt} ] && echo "Warning: expected ${n} data files, found ${n_tgt}" && exit 1;

    # (1a) Transcriptions and translations preparation
    # make basic transcription file (add segments info)
    cp ${yml} ${dst}/.yaml0
    grep duration ${dst}/.yaml0 > ${dst}/.yaml1
    awk '{
        duration=$3; offset=$5; spkid=$7;
        gsub(",","",duration);
        gsub(",","",offset);
        gsub(",","",spkid);
        duration=sprintf("%.7f", duration);
        if ( duration < 0.2 ) extendt=sprintf("%.7f", (0.2-duration)/2);
        else extendt=0;
        offset=sprintf("%.7f", offset);
        startt=offset-extendt;
        endt=offset+duration+extendt;
        printf("mtedx%s_%07.0f_%07.0f\n", spkid, int(1000*startt+0.5), int(1000*endt+0.5));
    }' ${dst}/.yaml1 > ${dst}/.yaml2
    # NOTE: Extend the lengths of short utterances (< 0.2s) rather than exclude them
    # NOTE: adding prefix "mtedx" is important for avoiding an error occured during speed perturbation

    cp ${src_text} ${dst}/${src_lang}.org
    cp ${tgt_text} ${dst}/${tgt_lang}.org

    for lang in ${src_lang} ${tgt_lang}; do
        # normalize punctuation
        normalize-punctuation.perl -l ${lang} < ${dst}/${lang}.org > ${dst}/${lang}.norm

        # lowercasing
        lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
        cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

        # remove punctuation
        remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm

        for case in lc.rm lc tc; do
            # tokenization
            tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.${case} > ${dst}/${lang}.norm.${case}.tok

            # paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.${case}.tok | sort > ${dst}/text.${case}.${lang}
            paste -d " " <(cat ${dst}/.yaml2) <(cat ${dst}/${lang}.norm.${case}.tok | awk '{if(NF>0) {print $0;} else {print "emptyutterance";}}') \
                > ${dst}/text.${case}.${lang}
        done

        # save original and cleaned punctuation
        lowercase.perl < ${dst}/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
        lowercase.perl < ${dst}/${lang}.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
    done


    # error check
    n=$(cat ${dst}/.yaml2 | wc -l)
    n_src=$(cat ${dst}/${src_lang}.norm.tc.tok | wc -l)
    n_tgt=$(cat ${dst}/${tgt_lang}.norm.tc.tok | wc -l)
    [ ${n} -ne ${n_src} ] && echo "Warning: expected ${n} data files, found ${n_src}" && exit 1;
    [ ${n} -ne ${n_tgt} ] && echo "Warning: expected ${n} data files, found ${n_tgt}" && exit 1;


    # (1c) Make segments files from transcript
    #segments file format is: utt-id start-time end-time, e.g.:
    # 0u7tTptBo9I_0000 0u7tTptBo9I 17.22 17.69
    awk '{
        segment=$1; n=split(segment,S,"[_]");
        if ( n==3 ) {spkid=S[1]; startf=S[2]; endf=S[3];}
        else if ( n==4 ) {spkid=S[1] "_" S[2]; startf=S[3]; endf=S[4];}
        else if ( n==5 ) {spkid=S[1] "_" S[2] "_" S[3]; startf=S[4]; endf=S[5];}
        printf("%s %s %.2f %.2f\n", segment, spkid, startf/1000, endf/1000);
    }' < ${dst}/text.tc.${tgt_lang} | uniq | sort > ${dst}/segments

    awk '{
        segment=$1; n=split(segment,S,"[_]");
        if ( n==3 ) spkid=S[1];
        else if ( n==4 ) spkid=S[1] "_" S[2];
        else if ( n==5 ) spkid=S[1] "_" S[2] "_" S[3];
        spkid_fix=spkid;
        gsub("mtedx","",spkid_fix);
        # printf("%s flac -c -d -s '${wav_dir}'/%s.flac |\n", spkid, spkid_fix);
        # printf("%s ffmpeg -i '${wav_dir}'/%s.flac -f wav -ar 44100 -ab 16 -ac 1 - |\n", spkid, spkid_fix);
        printf("%s ffmpeg -i '${wav_dir}'/%s.flac -f wav -ar 44100 -ac 1 - |\n", spkid, spkid_fix);
    }' < ${dst}/text.tc.${tgt_lang} | uniq | sort > ${dst}/wav.scp

    awk '{
        segment=$1; n=split(segment,S,"[_]");
        if ( n==3 ) spkid=S[1];
        else if ( n==4 ) spkid=S[1] "_" S[2];
        else if ( n==5 ) spkid=S[1] "_" S[2] "_" S[3];
        print $1 " " spkid
    }' ${dst}/segments | uniq | sort > ${dst}/utt2spk

    cat ${dst}/utt2spk | utils/utt2spk_to_spk2utt.pl | sort > ${dst}/spk2utt

    # error check
    n_src=$(cat ${dst}/text.tc.${src_lang} | wc -l)
    n_tgt=$(cat ${dst}/text.tc.${tgt_lang} | wc -l)
    [ ${n_src} -ne ${n_tgt} ] && echo "Warning: expected ${n_src} data files, found ${n_tgt}" && exit 1;

    # Copy stuff into its final locations [this has been moved from the format_data script]
    mkdir -p data/${set}.${src_lang}-${tgt_lang}

    # remove duplicated utterances (the same offset)
    echo "remove duplicate lines..."
    cut -d ' ' -f 1 ${dst}/text.tc.${src_lang} | sort | uniq -c | sort -n -k1 -r | grep -v '   1' \
        | sed 's/^[ \t]*//' > ${dst}/duplicate_lines
    cut -d ' ' -f 1 ${dst}/text.tc.${src_lang} | sort | uniq -c | sort -n -k1 -r | grep '   1' \
        | cut -d '1' -f 2- | sed 's/^[ \t]*//' > ${dst}/reclist
    reduce_data_dir.sh ${dst} ${dst}/reclist data/${set}.${src_lang}-${tgt_lang}
    for case in lc.rm lc tc; do
        cp ${dst}/text.${case}.${src_lang} data/${set}.${src_lang}-${tgt_lang}
        cp ${dst}/text.${case}.${tgt_lang} data/${set}.${src_lang}-${tgt_lang}
    done
    utils/fix_data_dir.sh --utt_extra_files \
        "text.tc.${src_lang} text.lc.${src_lang} text.lc.rm.${src_lang} \
         text.tc.${tgt_lang} text.lc.${tgt_lang} text.lc.rm.${tgt_lang}" data/${set}.${src_lang}-${tgt_lang}

    # remove empty and short utterances
    cp -rf data/${set}.${src_lang}-${tgt_lang} data/${set}.${src_lang}-${tgt_lang}.tmp
    grep -v emptyutterance data/${set}.${src_lang}-${tgt_lang}/text.tc.${src_lang} | cut -f 1 -d " " | sort > data/${set}.${src_lang}-${tgt_lang}/reclist.${src_lang}
    grep -v emptyutterance data/${set}.${src_lang}-${tgt_lang}/text.tc.${tgt_lang} | cut -f 1 -d " " | sort > data/${set}.${src_lang}-${tgt_lang}/reclist.${tgt_lang}
    comm -12 data/${set}.${src_lang}-${tgt_lang}/reclist.${src_lang} data/${set}.${src_lang}-${tgt_lang}/reclist.${tgt_lang} > data/${set}.${src_lang}-${tgt_lang}/reclist
    reduce_data_dir.sh data/${set}.${src_lang}-${tgt_lang}.tmp data/${set}.${src_lang}-${tgt_lang}/reclist data/${set}.${src_lang}-${tgt_lang}
    utils/fix_data_dir.sh --utt_extra_files \
        "text.tc.${src_lang} text.lc.${src_lang} text.lc.rm.${src_lang} \
         text.tc.${tgt_lang} text.lc.${tgt_lang} text.lc.rm.${tgt_lang}" data/${set}.${src_lang}-${tgt_lang}
    rm -rf data/${set}.${src_lang}-${tgt_lang}.tmp

    # error check
    n_seg=$(cat data/${set}.${src_lang}-${tgt_lang}/segments | wc -l)
    n_text=$(cat data/${set}.${src_lang}-${tgt_lang}/text.tc.${tgt_lang} | wc -l)
    [ ${n_seg} -ne ${n_text} ] && echo "Warning: expected ${n_seg} data files, found ${n_text}" && exit 1;

    echo "$0: successfully prepared data in ${dst}"
done
