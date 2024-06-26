#!/usr/bin/env bash

# Copyright 2019-2024 Kyoto University (Hirofumi Inaguma, Shuichiro Shimizu)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# This script converts the IWSLT 2024 Indic track dataset into the format of ESPnet recipe.
# For ESPnet data format, see https://github.com/espnet/data_example/blob/main/README.md

export LC_ALL=C

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
    log "Usage: $0 <data_dir> <tgt_lang>"
    log "e.g.: $0 /path/to/indic/data hi"
    exit 1;
fi

data_dir=$1
tgt_lang=$2

for split in train dev; do
    src=${data_dir}/en-${tgt_lang}/data/${split}
    dst=data/local/en-${tgt_lang}/${split}

    [ ! -d ${src} ] && log "$0: no such directory ${src}" && exit 1;

    wav_dir=${src}/wav
    txt_dir=${src}/txt
    yaml=${txt_dir}/${split}.yaml
    en=${txt_dir}/${split}.en
    tgt=${txt_dir}/${split}.${tgt_lang}

    mkdir -p ${dst} || exit 1;

    [ ! -d ${wav_dir} ] && log "$0: no such directory ${wav_dir}" && exit 1;
    [ ! -d ${txt_dir} ] && log "$0: no such directory ${txt_dir}" && exit 1;
    [ ! -f ${yaml} ] && log "$0: expected file ${yaml} to exist" && exit 1;
    [ ! -f ${en} ] && log "$0: expected file ${en} to exist" && exit 1;
    [ ! -f ${tgt} ] && log "$0: expected file ${tgt} to exist" && exit 1;

    wav_scp=${dst}/wav.scp; [[ -f "${wav_scp}" ]] && rm ${wav_scp}
    trans_en=${dst}/text.en; [[ -f "${trans_en}" ]] && rm ${trans_en}
    trans_tgt=${dst}/text.${tgt_lang}; [[ -f "${trans_tgt}" ]] && rm ${trans_tgt}
    utt2spk=${dst}/utt2spk; [[ -f "${utt2spk}" ]] && rm ${utt2spk}
    spk2utt=${dst}/spk2utt; [[ -f "${spk2utt}" ]] && rm ${spk2utt}
    segments=${dst}/segments; [[ -f "${segments}" ]] && rm ${segments}

    # error check
    n=$(cat ${yaml} | grep duration | wc -l)
    n_en=$(cat ${en} | wc -l)
    n_tgt=$(cat ${tgt} | wc -l)
    [ ${n} -ne ${n_en} ] && log "Error: expected ${n} data entries, found ${n_en}" && exit 1;
    [ ${n} -ne ${n_tgt} ] && log "Error: expected ${n} data entries, found ${n_tgt}" && exit 1;

	# copy files to ${dst}, removing empty lines
	cp ${yaml} ${dst}/.yaml0
	cp ${en} ${dst}/en.org
	cp ${tgt} ${dst}/${tgt_lang}.org

	empty_lines=$(grep -n '^$' "${yaml}" "${en}" "${tgt}" | cut -d ':' -f 2 | sort -nu | tr '\n' ',')
	sed_commands=$(echo "${empty_lines}" | sed 's/,/d;/g')

	if [ -z "$sed_commands" ]; then
		log "No empty lines found in ${src}"
	else
		sed -i -e "${sed_commands}" "${dst}/.yaml0" "${dst}/en.org" "${dst}/${tgt_lang}.org"
		log "Found empty lines at line ${empty_lines} in ${src}, removing them for further processing. The original files are kept in ${src}"
	fi

    # transcriptions and translations text file preparation
    grep duration ${dst}/.yaml0 > ${dst}/.yaml1

    # make utt_id from yaml
    # e.g., - {duration: 3.079999, offset: 7.28, speaker_id: spk.4, wav: bn4.wav} -> ted_00004_0007280_0010360
    # NOTE: Extend the lengths of short utterances (< 0.2s) rather than exclude them
    awk '{
        duration=$3; offset=$5; spkid=$7;
        gsub(",","",duration);
        gsub(",","",offset);
        gsub(",","",spkid);
        gsub("spk.","",spkid);
        duration=sprintf("%.7f", duration);
        if ( duration < 0.2 ) extendt=sprintf("%.7f", (0.2-duration)/2);
        else extendt=0;
        offset=sprintf("%.7f", offset);
        startt=offset-extendt;
        endt=offset+duration+extendt;
        printf("ted_%05d_%07.0f_%07.0f\n", spkid, int(1000*startt+0.5), int(1000*endt+0.5));
    }' ${dst}/.yaml1 > ${dst}/.yaml2

    # text normalization
    for lang in en ${tgt_lang}; do
        # normalize punctuation
        normalize-punctuation.perl -l ${lang} < ${dst}/${lang}.org > ${dst}/${lang}.norm

        # lowercasing
        lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
        cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

        # remove punctuation
        ../../../utils/remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm

        # tokenization
        tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.tc > ${dst}/${lang}.norm.tc.tok
        tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.tok
        tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc.rm > ${dst}/${lang}.norm.lc.rm.tok

        paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.tc.tok | sort > ${dst}/text.tc.${lang}
        paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.lc.tok | sort > ${dst}/text.lc.${lang}
        paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.lc.rm.tok | sort > ${dst}/text.lc.rm.${lang}

        # save original and cleaned punctuation
        lowercase.perl < ${dst}/${lang}.org | ../../../utils/text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
        lowercase.perl < ${dst}/${lang}.norm.tc | ../../../utils/text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
    done

    # error check
    n=$(cat ${dst}/.yaml2 | wc -l)
    n_en=$(cat ${dst}/en.norm.tc.tok | wc -l)
    n_tgt=$(cat ${dst}/${tgt_lang}.norm.tc.tok | wc -l)
    [ ${n} -ne ${n_en} ] && log "Error: expected ${n} data entries, found ${n_en}" && exit 1;
    [ ${n} -ne ${n_tgt} ] && log "Error: expected ${n} data entries, found ${n_tgt}" && exit 1;

    # segments file preparation
    awk '{
        segment=$1; split(segment,S,"[_]");
        spkid=S[1] "_" S[2]; startf=S[3]; endf=S[4];
        printf("%s %s %.2f %.2f\n", segment, spkid, startf/1000, endf/1000);
    }' < ${dst}/text.tc.${tgt_lang} | uniq | sort > ${dst}/segments

    # wav.scp file preparation
    awk '{
        segment=$1; split(segment,S,"[_]");
        spkid=S[1] "_" S[2];
        printf("%s cat '${wav_dir}'/'${tgt_lang}'%d.wav |\n", spkid, S[2]);
    }' < ${dst}/text.tc.${tgt_lang} | uniq | sort > ${dst}/wav.scp

    # utt2spk file preparation
    awk '{
        segment=$1; split(segment,S,"[_]");
        spkid=S[1] "_" S[2]; print $1 " " spkid
    }' ${dst}/segments | uniq | sort > ${dst}/utt2spk

    # spk2utt file preparation
    cat ${dst}/utt2spk | utils/utt2spk_to_spk2utt.pl | sort > ${dst}/spk2utt

    # error check
    n_en=$(cat ${dst}/text.tc.en | wc -l)
    n_tgt=$(cat ${dst}/text.tc.${tgt_lang} | wc -l)
    [ ${n_en} -ne ${n_tgt} ] && log "Error: expected ${n_en} data entries, found ${n_tgt}" && exit 1;

    # copy files into its final locations
    mkdir -p data/${split}.en-${tgt_lang}

    # remove duplicated utterances (i.e. utterances with the same offset)
    log "removing duplicate lines..."
    cut -d ' ' -f 1 ${dst}/text.tc.en | sort | uniq -c | sort -n -k1 -r | grep -v '1 ted' \
        | sed 's/^[ \t]*//' > ${dst}/duplicate_lines
    cut -d ' ' -f 1 ${dst}/text.tc.en | sort | uniq -c | sort -n -k1 -r | grep '1 ted' \
        | cut -d '1' -f 2- | sed 's/^[ \t]*//' > ${dst}/reclist
    ../../../utils/reduce_data_dir.sh ${dst} ${dst}/reclist data/${split}.en-${tgt_lang}
    for l in en ${tgt_lang}; do
        for case in tc lc lc.rm; do
            cp ${dst}/text.${case}.${l} data/${split}.en-${tgt_lang}/text.${case}.${l}
        done
    done
    utils/fix_data_dir.sh --utt_extra_files \
        "text.tc.en text.lc.en text.lc.rm.en text.tc.${tgt_lang} text.lc.${tgt_lang} text.lc.rm.${tgt_lang}" \
        data/${split}.en-${tgt_lang}

    # error check
    n_seg=$(cat data/${split}.en-${tgt_lang}/segments | wc -l)
    n_text=$(cat data/${split}.en-${tgt_lang}/text.tc.${tgt_lang} | wc -l)
    [ ${n_seg} -ne ${n_text} ] && log "Error: expected ${n_seg} data entries, found ${n_text}" && exit 1;
done

for split in tst-COMMON; do
    src=${data_dir}/en-${tgt_lang}/data/${split}
    dst=data/local/en-${tgt_lang}/${split}

    [ ! -d ${src} ] && log "$0: no such directory ${src}" && exit 1;

    wav_dir=${src}/wav
    txt_dir=${src}/txt
    yaml=${txt_dir}/${split}.yaml

    mkdir -p ${dst} || exit 1;

    [ ! -d ${wav_dir} ] && log "$0: no such directory ${wav_dir}" && exit 1;
    [ ! -d ${txt_dir} ] && log "$0: no such directory ${txt_dir}" && exit 1;
    [ ! -f ${yaml} ] && log "$0: expected file ${yaml} to exist" && exit 1;

    wav_scp=${dst}/wav.scp; [[ -f "${wav_scp}" ]] && rm ${wav_scp}
    utt2spk=${dst}/utt2spk; [[ -f "${utt2spk}" ]] && rm ${utt2spk}
    spk2utt=${dst}/spk2utt; [[ -f "${spk2utt}" ]] && rm ${spk2utt}
    segments=${dst}/segments; [[ -f "${segments}" ]] && rm ${segments}

    cp ${yaml} ${dst}/.yaml0

    # make utt_id from yaml
    # e.g., - {duration: 3.079999, offset: 7.28, speaker_id: spk.4, wav: bn4.wav} -> ted_00004_0007280_0010360
    awk '{
        duration=$3; offset=$5; spkid=$7;
        gsub(",","",duration);
        gsub(",","",offset);
        gsub(",","",spkid);
        gsub("spk.","",spkid);
        duration=sprintf("%.7f", duration);
        offset=sprintf("%.7f", offset);
        startt=offset;
        endt=offset+duration;
        printf("ted_%05d_%07.0f_%07.0f\n", spkid, int(1000*startt+0.5), int(1000*endt+0.5));
    }' ${dst}/.yaml0 > ${dst}/.yaml2

    # segments file preparation
    awk '{
        segment=$1; split(segment,S,"[_]");
        spkid=S[1] "_" S[2]; startf=S[3]; endf=S[4];
        printf("%s %s %.2f %.2f\n", segment, spkid, startf/1000, endf/1000);
    }' < ${dst}/.yaml2 | uniq | sort > ${dst}/segments

    # wav.scp file preparation
    awk '{
        segment=$1; split(segment,S,"[_]");
        spkid=S[1] "_" S[2];
        printf("%s cat '${wav_dir}'/'${tgt_lang}'%d.wav |\n", spkid, S[2]);
    }' < ${dst}/.yaml2 | uniq | sort > ${dst}/wav.scp

    # utt2spk file preparation
    awk '{
        segment=$1; split(segment,S,"[_]");
        spkid=S[1] "_" S[2]; print $1 " " spkid
    }' ${dst}/segments | uniq | sort > ${dst}/utt2spk

    # spk2utt file preparation
    cat ${dst}/utt2spk | utils/utt2spk_to_spk2utt.pl | sort > ${dst}/spk2utt

    # copy files into its final locations
    final_dst=data/${split}.en-${tgt_lang}
    mkdir -p ${final_dst}
    cp ${dst}/segments ${final_dst}/segments
    cp ${dst}/wav.scp ${final_dst}/wav.scp
    cp ${dst}/utt2spk ${final_dst}/utt2spk
    cp ${dst}/spk2utt ${final_dst}/spk2utt

    # check if the files in the data directory are in correct format
    utils/fix_data_dir.sh ${final_dst}

    # error check
    n=$(cat ${dst}/.yaml2 | wc -l)
    n_seg=$(cat ${final_dst}/segments | wc -l)
    [ ${n} -ne ${n_seg} ] && log "Error: expected ${n} data entries, found ${n_seg}" && exit 1;
done

log "$0: successfully prepared data in ${dst}"
