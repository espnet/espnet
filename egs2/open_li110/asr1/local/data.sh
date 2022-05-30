#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

shopt -s extglob

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
'''langs="ab af bg cv eo fy-NL hu ka lv mt pl sah\
 st tn ta xh uk zh-HK ar bn cy es ga-IE hy-AM kab\
 mdf myv pt sat th ur zh-TW as br da et gl ia kk\
 mhr nan-tw ne rm-sursilv su sk tig uz az ca de eu\
 gn gu id kmr km kn mk nl rm-vallader sl tok te vi\
 ba ckb dv fa ha ig ky yo ml nn-NO ro sr tr vot bas\
 cnh el fi hi it lg mn or ru sv-SE tt yue be cs en\
 fr hsb ja jv lt mr pa-IN rw sw ug zh-CN si af in hr sv"
'''
langs="sat th ur zh-TW as br da et gl ia kk\
 mhr nan-tw ne rm-sursilv su sk tig uz az ca de eu\
 gn gu id kmr km kn mk nl rm-vallader sl tok te vi\
 ba ckb dv fa ha ig ky yo ml nn-NO ro sr tr vot bas\
 cnh el fi hi it lg mn or ru sv-SE tt yue be cs en\
 fr hsb ja jv lt mr pa-IN rw sw ug zh-CN si af in hr sv"
lid=true
nlsyms_txt=data/local/nlsyms.txt


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. utils/parse_options.sh

langs=$(echo "${langs}" | tr _ " ")
voxforge_lang="de en es fr it nl pt ru"
commonvoice_lang="ab bg cv eo fy-NL hu ka lv mt pl sah ta uk zh-HK ar bn cy es ga-IE hy-AM kab mdf myv pt sat th ur zh-TW as br da et gl ia kk mhr nan-tw rm-sursilv sk tig uz az ca de eu gn id kmr mk nl rm-vallader sl tok vi ba ckb dv fa ha ig ky ml nn-NO ro sr tr vot bas cnh el fi hi it lg mn or ru sv-SE tt yue be cs en fr hsb ja lt mr pa-IN rw sw ug zh-CN"
googlei18n_lang_30_32_37="si af st tn xh bn"
googlei18n_lang_asr="jv su si bn ne"
googlei18n_lang_tts="jv km ne su es ml mr ta te ca en es es es es es eu gl gu kn yo"
mls_lang="en de fr it es nl pl pt"
voxpopuli_lang="in en de fr es pl it ro hu cs nl fi hr sk sl et lt pt bg el lv mt sv or da"

train_set=train_li110_lid
train_dev=dev_li110_lid
test_set=

log "data preparation started"

mkdir -p ${COMMONVOICE}
mkdir -p ${VOXFORGE}
mkdir -p ${MLS}
mkdir -p ${GOOGLEI18N}
mkdir -p ${VOXPOPULI}

for lang in ${langs}; do

    if [[ "${commonvoice_lang}" == *"${lang}"* ]]; then
        if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
            log "sub-stage 0: Download Data to ${COMMONVOICE}"

            # base url for downloads.
            data_url=https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-9.0-2022-04-27/cv-corpus-9.0-2022-04-27-${lang}.tar.gz

            # local/voxforge/download_and_untar.sh ${COMMONVOICE} ${data_url} ${lang}.tar.gz
            # rm -f ${COMMONVOICE}/${lang}.tar.gz
        fi

        train_subset=train_"$(echo "${lang}" | tr - _)"_commonvoice
        train_subdev=dev_"$(echo "${lang}" | tr - _)"_commonvoice
        test_subset=test_"$(echo "${lang}" | tr - _)"_commonvoice

        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            log "sub-stage 1: Preparing Data for Commonvoice"

            if [ "${lang}" = sat ]; then
                 sets="validated test"
            else
                 sets="validated test dev"
            fi

            for part in ${sets}; do
                # use underscore-separated names in data directories.
                local/commonvoice/data_prep.pl "${COMMONVOICE}/cv-corpus-9.0-2022-04-27/${lang}" ${part} data/"$(echo "${part}_${lang}_commonvoice" | tr - _)" "${lang}_commonvoice"
            done
    
            # remove test&dev data from validated sentences
            utils/copy_data_dir.sh data/"$(echo "validated_${lang}_commonvoice" | tr - _)" data/${train_subset}
            if [ "${lang}" != sat ]; then
                utils/filter_scp.pl --exclude data/${train_subdev}/wav.scp data/${train_subset}/wav.scp > data/${train_subset}/temp_wav.scp
                utils/filter_scp.pl --exclude data/${test_subset}/wav.scp data/${train_subset}/temp_wav.scp > data/${train_subset}/wav.scp
                utils/fix_data_dir.sh data/${train_subset}
            else
                utils/filter_scp.pl --exclude data/${test_subset}/wav.scp data/${train_subset}/wav.scp > data/${train_subset}/temp_wav.scp
                mv data/${train_subset}/temp_wav.scp data/${train_subset}/wav.scp
                utils/fix_data_dir.sh data/${train_subset}
            fi
        fi
        test_set="${test_set} ${test_subset}"
    fi

    if [[ "${voxforge_lang}" == *"${lang}"* ]]; then
        if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
            log "sub-stage0: Download data to ${VOXFORGE}"

            if [ ! -e "${VOXFORGE}/${lang}/extracted" ]; then
                log "sub-stage 1: Download data to ${VOXFORGE}"
                local/voxforge/getdata.sh "${lang}" "${VOXFORGE}"
            else
                log "sub-stage 1: ${VOXFORGE}/${lang}/extracted is already existing. Skip data downloading"
            fi
        fi

        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            log "sub-stage 1: Data Preparation for Voxforge"
            selected=${VOXFORGE}/${lang}/extracted
            # Initial normalization of the data
            local/voxforge/voxforge_data_prep.sh --flac2wav false "${selected}" "${lang}"
            local/voxforge/voxforge_format_data.sh "${lang}"
	    utils/copy_data_dir.sh --utt-suffix -${lang}_voxforge data/all_"${lang}" data/validated_"${lang}"_voxforge
	    rm -r data/all_${lang}
            # following split consider prompt duplication (but does not consider speaker overlap instead)
            local/voxforge/split_tr_dt_et.sh data/validated_"${lang}"_voxforge data/train_"${lang}"_voxforge data/dev_"${lang}"_voxforge data/test_"${lang}"_voxforge
        fi

        test_set="${test_set} test_${lang}_voxforge"

    fi

    if [[ "$mls_lang}" == *"${lang}"* ]]; then
        # Get lang's name from its id for download links
        case ${lang} in
            "es")
                download_id=spanish ;;
            "en")
                download_id=english ;;
            "fr")
                download_id=french ;;
            "nl")
                download_id=dutch ;;
            "it")
                download_id=italian ;;
            "pt")
                download_id=portuguese ;;
            "pl")
                download_id=polish ;;
        esac 
        data_url=https://dl.fbaipublicfiles.com/mls/mls_${download_id}.tar.gz
        lm_data_url=https://dl.fbaipublicfiles.com/mls/mls_lm_${download_id}.tar.gz

        if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
            log "sub-stage 0: Download Data to ${MLS}"
        
            # local/mls/download_and_untar.sh ${MLS} ${data_url} mls_${download_id}.tar.gz
            # local/mls/download_and_untar.sh ${MLS} ${lm_data_url} mls_lm_${download_id}.tar.gz
            # Optional: mls corpus is large. You might want to remove them after processing
            # rm -f ${MLS}/mls_${download_id}.tar.gz
            # rm -f ${MLS}/mls_lm_${download_id}.tar.gz
        fi
        
        
        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            log "sub-stage 1: Preparing Data for MLS"
        
            python local/mls/data_prep.py --source ${MLS}/mls_${download_id} --lang ${lang} --prefix "mls_"
            utils/fix_data_dir.sh data/train_${lang}_mls
            utils/fix_data_dir.sh data/dev_${lang}_mls
            utils/fix_data_dir.sh data/test_${lang}_mls
        
            # add placeholder to align format with other corpora
            sed -r '/^\s*$/d' ${MLS}/mls_lm_${download_id}/data.txt | \
                 awk '{printf("%.8d %s\n"), NR-1, $0}'  >> data/lm_train.txt
        fi
        
        test_set="${test_set} test_${lang}_mls"
    fi

    if [[ "${googlei18n_lang_tts}" == *"${lang}"* ]]; then
        case ${lang} in
            "es")
                processing_list="61-es_ar 71-es_cl 72-es_co 73-es_pe 75-es_ve" ;;
            "en")
                processing_list="70-en_ng 83-midlands_english 83-northern_english 83-scottish_english 83-southern_english 83-welsh_english" ;;
            "ml")
                processing_list="63-ml_in" ;;
            "ta")
                processing_list="65-ta_in" ;;
            "te")
                processing_list="66-te_in" ;;
            "ca")
                processing_list="69-ca_es" ;;
            "eu")
                processing_list="76-eu_es" ;;
            "gl")
                processing_list="77-gl_es" ;;
            "gu")
                processing_list="78-gu_in" ;;
            "kn")
                processing_list="79-kn_in" ;;
            "yo")
                processing_list="86-yo_ng" ;;
        esac

        case ${lang} in
            "mr")
                female_processing_list="64-mr_in" ;;
            "es")
                female_processing_list="74-es_pr" ;;
            "my")
                female_processing_list="80-my_mm" ;;
        esac

        case ${lang} in
            "en")
                male_processing_list="83-irish_english" ;;
        esac

        case ${lang} in
            "jv")
                v1_processing_list="41-jv_id-male 41-jv_id-female" ;;
            "km")
                v1_processing_list="42-km_kh-male" ;;
            "ne")
                v1_processing_list="43-ne_np-female" ;;
            "su")
                v1_processing_list="44-su_id-female 44-su_id-male" ;;
        esac

        if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
            log "sub-stage0: Download data to ${GOOGLEI18N}"
            # TODO(Jiatong): add download procedure, pending due to inconsistent structure in openslr
            processing_list=
            female_processing_list=
            male_processing_list=
            v1_processing_list=
            for dialect_lang in ${processing_list}; do
                echo "pending download both procedure"
                openslr_id=$(echo ${dialect_lang} | cut -f1 -d-)
                lang_id=$(echo ${dialect_lang} | cut -f2 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}
                cat ${data_folder}/line_index_male.tsv ${data_folder}/line_index_female.tsv > ${data_folder}/line_index.tsv
            done

            for dialect_lang in ${female_processing_list}; do
                echo "pending download female procedure"
                openslr_id=$(echo ${dialect_lang} | cut -f1 -d-)
                lang_id=$(echo ${dialect_lang} | cut -f2 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}
                cat ${data_folder}/line_index_female.tsv > ${data_folder}/line_index.tsv
            done

            for dialect_lang in ${male_processing_list}; do
                echo "pending download male procedure"
                openslr_id=$(echo ${dialect_lang} | cut -f1 -d-)
                lang_id=$(echo ${dialect_lang} | cut -f2 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}
                cat ${data_folder}/line_index_male.tsv > ${data_folder}/line_index.tsv
            done

            for dialect_lang in ${v1_processing_list}; do
                echo "pending download v1 procedure"
                openslr_id=$(echo ${dialect_lang} | cut -f1 -d-)
                lang_id=$(echo ${dialect_lang} | cut -f2 -d-)
                gender=$(echo ${dialect_lang} | cut -f3 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}/${lang_id}_${gender}
                # cat ${data_folder}/line_index_male.tsv ${data_folder}/line_index_female.tsv > ${data_folder}/line_index.tsv
            done
             
        fi

        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            log "sub-stage 1: Data Preparation for Googlei18n TTS"
            for dialect_lang in ${processing_list} ${female_processing_list} ${male_processing_list}; do
                echo "pending download both procedure"
                openslr_id=$(echo ${dialect_lang} | cut -f1 -d-)
                lang_id=$(echo ${dialect_lang} | cut -f2 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}
                target_dir=data/openslr${openslr_id}_${lang_id}
                awk -v lang_id="${lang_id}" -F '[_\t]' '{print "openslr" lang_id "_" $1 "_" $2 "_" $3 " " "openslr" lang_id "_" $1 "_" $2}' ${data_folder}/line_index.tsv  > ${target_dir}/utt2spk
                awk -v lang_id="${lang_id}" -F '[_\t]' '{print "openslr" lang_id "_" $1 "_" $2 "_" $3 " " $5}' ${data_folder}/line_index.tsv  > ${target_dir}/text
                awk -v lang_id="${lang_id}" -v src_dir="${data_folder}" -F '[_\t]' '{print "openslr" lang_id "_" $1 "_" $2 "_" $3 " " src_dir "/" $1 "_" $2 "_" $3 ".wav"' ${data_folder}/line_index.tsv  > ${target_dir}/wav.scp
                sort ${target_dir}/utt2spk -o ${target_dir}/utt2spk
                sort ${target_dir}/text -o ${target_dir}/text
                sort ${target_dir}/wav.scp -o ${target_dir}/wav.scp
                utils/utt2spk_to_spk2utt.pl ${target_dir}/utt2spk > ${target_dir}/spk2utt
                utils/validate_data_dir.sh --no_feats ${target_dir}

                utils/subset_data_dir.sh data/openslr${openslr_id}_${lang_id} 500 data/dev-test-${lang}-openslr${openslr_id}
                utils/subset_data_dir.sh data/dev-test-${lang}-openslr${openslr_id} 250 data/dev_${lang}_openslr${openslr_id}
                utils/copy_data_dir.sh data/dev-test-${lang}-openslr${openslr_id} data/test_${lang}_openslr${openslr_id}
                utils/filter_scp.pl --exclude data/dev_${lang}_openslr${openslr_id}/wav.scp \
                    data/dev-test-${lang}-openslr${openslr_id}/wav.scp > data/test_${lang}_openslr${openslr_id}/wav.scp
                utils/fix_data_dir.sh data/test_${lang}_openslr${openslr_id}

                utils/copy_data_dir.sh data/openslr${openslr_id}_${lang_id} data/train_${lang}_openslr${openslr_id}
                utils/filter_scp.pl --exclude data/dev-test-${lang}_openslr${openslr_id}/wav.scp \
                    data/${lang}-openslr${openslr_id}/wav.scp > data/train_${lang}_openslr${openslr_id}/wav.scp
                utils/fix_data_dir.sh data/train_${lang}_openslr${openslr_id}/wav.scp
                test_set="${test_set} test_${lang}_openslr${openslr_id}"                
            done

            for dialect_lang in ${v1_processing_list}; do
                echo "pending download v1 procedure"
                openslr_id=$(echo ${dialect_lang} | cut -f1 -d-)
                lang_id=$(echo ${dialect_lang} | cut -f2 -d-)
                gender=$(echo ${dialect_lang} | cut -f3 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}/${lang_id}_${gender}
                target_dir=data/openslr${openslr_id}_${lang_id}_${gender}
                awk -v lang_id="${lang_id}" -F '[_\t]' '{print "openslr" lang_id "_" $1 "_" $2 "_" $3 " " "openslr" lang_id "_" $1 "_" $2}' ${data_folder}/line_index.tsv  > ${target_dir}/utt2spk
                awk -v lang_id="${lang_id}" -F '[_\t]' '{print "openslr" lang_id "_" $1 "_" $2 "_" $3 " " $5}' ${data_folder}/line_index.tsv  > ${target_dir}/text
                awk -v lang_id="${lang_id}" -v src_dir="${data_folder}" -F '[_\t]' '{print "openslr" lang_id "_" $1 "_" $2 "_" $3 " " src_dir "/" $1 "_" $2 "_" $3 ".wav"' ${data_folder}/line_index.tsv  > ${target_dir}/wav.scp
                sort ${target_dir}/utt2spk -o ${target_dir}/utt2spk
                sort ${target_dir}/text -o ${target_dir}/text
                sort ${target_dir}/wav.scp -o ${target_dir}/wav.scp
                utils/utt2spk_to_spk2utt.pl ${target_dir}/utt2spk > ${target_dir}/spk2utt
                utils/validate_data_dir.sh --no_feats ${target_dir}

                utils/subset_data_dir.sh data/openslr${openslr_id}_${lang_id}_${gender} 500 data/dev-test-${lang}-openslr${openslr_id}_${gender}
                utils/subset_data_dir.sh data/dev-test-${lang}-openslr${openslr_id} 250 data/dev_${lang}_openslr${openslr_id}_${gender}
                utils/copy_data_dir.sh data/dev-test-${lang}-openslr${openslr_id}_${gender} data/test_${lang}_openslr${openslr_id}_${gender}
                utils/filter_scp.pl --exclude data/dev_${lang}_openslr${openslr_id}_${gender}/wav.scp \
                    data/dev-test-${lang}-openslr${openslr_id}_${gender}/wav.scp > data/test_${lang}_openslr${openslr_id}_${gender}/wav.scp
                utils/fix_data_dir.sh data/test_${lang}_openslr${openslr_id}_${gender}

                utils/copy_data_dir.sh data/openslr${openslr_id}_${lang_id}_${gender} data/train_${lang}_openslr${openslr_id}
                utils/filter_scp.pl --exclude data/dev-test-${lang}_openslr${openslr_id}_${gender}/wav.scp \
                    data/${lang}-openslr${openslr_id}_${gender}/wav.scp > data/train_${lang}_openslr${openslr_id}_${gender}/wav.scp
                utils/fix_data_dir.sh data/train_${lang}_openslr${openslr_id}_${gender}/wav.scp
                test_set="${test_set} test_${lang}_openslr${openslr_id}_${gender}"
            done 
        fi

    fi


done

log "Using test sets: ${test_set}"

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Combine Datadir"

    utils/combine_data.sh --skip_fix true data/train_temp data/train_!(*temp|*li110_*)
    utils/combine_data.sh --skip_fix true data/dev_temp data/dev_!(*temp|*li110_*)

    # Perform text preprocessing (upper case, remove punctuation)
    # Original text: 
    #     But, most important, he was able every day to live out his dream. 
    #     "Ask me why; I know why."
    # --->
    # Upper text:
    #     BUT, MOST IMPORTANT, HE WAS ABLE EVERY DAY TO LIVE OUT HIS DREAM.
    #     "ASK ME WHY; I KNOW WHY."
    # ---->
    # Punctuation remove: 
    #     BUT MOST IMPORTANT HE WAS ABLE EVERY DAY TO LIVE OUT HIS DREAM
    #     ASK ME WHY I KNOW WHY

    for x in data/train_temp data/dev_temp; do
        cp ${x}/text ${x}/text.org
        paste -d " " \
              <(cut -f 1 -d" " ${x}/text.org) \
              <(cut -f 2- -d" " ${x}/text.org \
                | python3 -c 'import sys; print(sys.stdin.read().upper(), end="")' \
                | python3 -c 'import string; print(sys.stdin.read().translate(str.maketrans("", "", string.punctuation)), end="")') \
              > ${x}/text
        rm ${x}/text.org
    done

    for x in ${test_set}; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " \
              <(cut -f 1 -d" " data/${x}/text.org) \
              <(cut -f 2- -d" " data/${x}/text.org \
                | python3 -c 'import sys; print(sys.stdin.read().upper(), end="")' \
                | python3 -c 'import string; print(sys.stdin.read().translate(str.maketrans("", "", string.punctuation)), end="")') \
              > data/${x}/text
        rm data/${x}/text.org
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Add Language ID"

    cp -r data/train_temp data/${train_set}
    cp -r data/dev_temp data/${train_dev}

    if [ "$lid" = true ]
    then

        # Original text: 
        #     BUT MOST IMPORTANT HE WAS ABLE EVERY DAY TO LIVE OUT HIS DREAM
        #     ASK ME WHY I KNOW WHY
        # --->
        # Add language ID: 
        #     [en] BUT MOST IMPORTANT HE WAS ABLE EVERY DAY TO LIVE OUT HIS DREAM
        #     [en] ASK ME WHY I KNOW WHY

        paste -d " " \
       <(cut -f 1 -d" " data/train_temp/text) \
       <(cut -f 1 -d" " data/train_temp/text | sed -e "s/.*\-\(.*\)_.*/\1/" | sed -e "s/_[^TW]\+//" | sed -e "s/^/\[/" -e "s/$/\]/") \
       <(cut -f 2- -d" " data/train_temp/text) | sed -e "s/\([^[]*\[[^]]*\]\)\s\(.*\)/\1\2/" \
       > data/${train_set}/text
        paste -d " " \
       <(cut -f 1 -d" " data/dev_temp/text) \
       <(cut -f 1 -d" " data/dev_temp/text | sed -e "s/.*\-\(.*\)_.*/\1/" | sed -e "s/_[^TW]\+//" | sed -e "s/^/\[/" -e "s/$/\]/") \
       <(cut -f 2- -d" " data/dev_temp/text) | sed -e "s/\([^[]*\[[^]]*\]\)\s\(.*\)/\1\2/" \
       > data/${train_dev}/text

        new_test_set=""
        for x in ${test_set}; do
            cp -r data/${x} data/${x}_lid
           paste -d " " \
           <(cut -f 1 -d" " data/${x}/text) \
           <(cut -f 1 -d" " data/${x}/text | sed -e "s/.*\-\(.*\)_.*/\1/" | sed -e "s/_[^TW]\+//" | sed -e "s/^/\[/" -e "s/$/\]/") \
           <(cut -f 2- -d" " data/${x}/text) | sed -e "s/\([^[]*\[[^]]*\]\)\s\(.*\)/\1\2/" \
           > data/${x}_lid/text
           new_test_set="${new_test_set} ${x}_lid"
        done
        echo "test set are saved as ${new_test_set}"

    fi

    utils/fix_data_dir.sh data/${train_set}
    utils/fix_data_dir.sh data/${train_dev}

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Create Non-linguistic Symbols for Language ID"
    cut -f 2- data/${train_set}/text | grep -o -P '\[.*?\]|\<.*?\>' | sort | uniq > ${nlsyms_txt}
    log "save non-linguistic symbols in ${nlsyms_txt}"
fi



log "Successfully finished. [elapsed=${SECONDS}s]"
