#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

shopt -s extglob

# general configuration
stage=2       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
# missing tig because of only 6 utt in cv
langs="ab af bg cv eo fy-NL hu ka lv mt pl sah\
 st tn ta xh uk zh-HK ar bn cy es ga-IE hy-AM kab\
 mdf myv pt sat th ur zh-TW as br da et gl ia kk\
 mhr nan-tw ne su sk uz az ca de eu\
 gn gu id km kn mk nl sl tok te vi\
 ba ckb dv fa ha ig ky yo ml nn-NO ro sr tr vot bas\
 cnh el fi hi it lg mn or ru sv-SE tt yue be cs en\
 fr hsb ja jv lt mr pa-IN rw sw ug zh-CN si af in hr sv"
extra_langs="rm-sursilv rm-vallader kmr mhr sv-SE" # for cv only
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
commonvoice_lang="ab bg cv eo fy-NL hu ka lv mt pl sah ta uk zh-HK ar bn cy es ga-IE hy-AM kab mdf myv pt sat th ur zh-TW as br da et gl ia kk nan-tw sk uz az ca de eu gn id mk nl sl tok vi ba ckb dv fa ha ig ky ml nn-NO ro sr tr vot bas cnh el fi hi it lg mn or ru tt yue be cs fr hsb ja lt pa-IN rw sw ug zh-CN"
googlei18n_lang_30_32_37="af st tn xh bn"
googlei18n_lang_asr="jv su si bn ne"
googlei18n_lang_tts="jv km ne su es ml mr ta te ca es es es es es eu gl gu kn yo"
mls_lang="en de fr it es nl pl pt"
voxpopuli_lang="en de fr es pl it ro hu cs nl fi hr sk sl et lt"

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

    if [[ "${commonvoice_lang}" == *" ${lang}"* || "${commonvoice_lang}" == *"${lang} "* ]]; then
        if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
            log "sub-stage 0: Download Data to ${COMMONVOICE}"

            # base url for downloads.
            data_url=https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-9.0-2022-04-27/cv-corpus-9.0-2022-04-27-${lang}.tar.gz

            local/voxforge/download_and_untar.sh ${COMMONVOICE} ${data_url} ${lang}.tar.gz
            # (Optional) remove archived file
            # rm -f ${COMMONVOICE}/${lang}.tar.gz
        fi

        train_subset=train_"$(echo "${lang}" | tr - _)"_commonvoice
        train_subdev=dev_"$(echo "${lang}" | tr - _)"_commonvoice
        test_subset=test_"$(echo "${lang}" | tr - _)"_commonvoice

        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            log "sub-stage 1: Preparing Data for Commonvoice"

            if [ "${lang}" = sat ] || \
               [ "${lang}" = mk ] || \
               [ "${lang}" = ha ] || \
               [ "${lang}" = ig ] || \
               [ "${lang}" = ml ] || \
               [ "${lang}" = vot ]; then
                 sets="validated test"
            else
                 sets="validated test dev"
            fi

            for part in ${sets}; do
                # use underscore-separated names in data directories.
                local/commonvoice/data_prep.pl "${COMMONVOICE}/cv-corpus-9.0-2022-04-27/${lang}" ${part} data/"$(echo "${part}_${lang}_commonvoice" | tr - _)" "${lang}_commonvoice"
            done
    
            # remove test&dev data from validated sentences
            utils/copy_data_dir.sh --validate_opts "--non-print" \
                data/"$(echo "validated_${lang}_commonvoice" | tr - _)" data/${train_subset}
            if [ "${lang}" != sat ] && \
               [ "${lang}" != mk ] && \
               [ "${lang}" != ha ] && \
               [ "${lang}" != ig ] && \
               [ "${lang}" != ml ] && \
               [ "${lang}" != vot ]; then
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
	    utils/copy_data_dir.sh --validate_opts "--non-print" --utt-suffix -${lang}_voxforge data/all_"${lang}" data/validated_"${lang}"_voxforge
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
            "de")
                download_id=german ;;
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
        
            local/mls/download_and_untar.sh ${MLS} ${data_url} mls_${download_id}.tar.gz
            local/mls/download_and_untar.sh ${MLS} ${lm_data_url} mls_lm_${download_id}.tar.gz
            # Optional: mls corpus is large. You might want to remove them after processing
            # rm -f ${MLS}/mls_${download_id}.tar.gz
            # rm -f ${MLS}/mls_lm_${download_id}.tar.gz
        fi
        
        
        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            log "sub-stage 1: Preparing Data for MLS"
            python local/mls/data_prep.py --source ${MLS}/mls_${download_id} --lang ${lang} --prefix "mls_" --suffix "\-${lang}_mls"
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
        processing_list=
        female_processing_list=
        male_processing_list=
        v1_processing_list=
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
            for dialect_lang in ${processing_list}; do
                echo "we do not provide download scripts now because of format inconsistency"
                echo "please download on your own"
                openslr_id=$(echo ${dialect_lang} | cut -f1 -d-)
                lang_id=$(echo ${dialect_lang} | cut -f2 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}
                cat ${data_folder}/line_index_male.tsv ${data_folder}/line_index_female.tsv > ${data_folder}/line_index.tsv
            done

            for dialect_lang in ${female_processing_list}; do
                echo "we do not provide download scripts now because of format inconsistency"
                echo "please download on your own"
                openslr_id=$(echo ${dialect_lang} | cut -f1 -d-)
                lang_id=$(echo ${dialect_lang} | cut -f2 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}
                cat ${data_folder}/line_index_female.tsv > ${data_folder}/line_index.tsv
            done

            for dialect_lang in ${male_processing_list}; do
                echo "we do not provide download scripts now because of format inconsistency"
                echo "please download on your own"
                openslr_id=$(echo ${dialect_lang} | cut -f1 -d-)
                lang_id=$(echo ${dialect_lang} | cut -f2 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}
                cat ${data_folder}/line_index_male.tsv > ${data_folder}/line_index.tsv
            done

            for dialect_lang in ${v1_processing_list}; do
                echo "we do not provide download scripts now because of format inconsistency"
                echo "please download on your own"
                openslr_id=$(echo ${dialect_lang} | cut -f1 -d-)
                lang_id=$(echo ${dialect_lang} | cut -f2 -d-)
                gender=$(echo ${dialect_lang} | cut -f3 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}/${lang_id}_${gender}
            done
             
        fi

        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            log "sub-stage 1: Data Preparation for Googlei18n TTS"

            for dialect_lang in ${processing_list} ${female_processing_list} ${male_processing_list}; do
                openslr_id=$(echo ${dialect_lang} | cut -f1 -d-)
                lang_id=$(echo ${dialect_lang} | cut -f2 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}
                target_dir=data/openslr${openslr_id}_${lang_id}
                mkdir -p ${target_dir}

                awk -v lang_id="${lang_id}" -F '[_\t]' \
                    '{print $1 "_" $2 "_" $3 "-" lang_id "_openslr " $1 "_" $2}' \
                    ${data_folder}/line_index.tsv  > ${target_dir}/utt2spk
                awk -v lang_id="${lang_id}" -F '[\t]' \
                    '{print $1 "-" lang_id "_openslr " $2}' \
                    ${data_folder}/line_index.tsv  > ${target_dir}/text
                if [ ${lang_id} = bn_in ]; then
                    awk -v lang_id="${lang_id}" -v src_dir="${data_folder}" -F '[_\t]' \
                        '{print $1 "_" $2 "_" $3 "-" lang_id "_openslr " src_dir "/" $1 "/" $1 "_" $2 "_" $3}' \
                        ${data_folder}/line_index.tsv  > ${target_dir}/wav.scp
                else
                    awk -v lang_id="${lang_id}" -v src_dir="${data_folder}" -F '[_\t]' \
                        '{print $1 "_" $2 "_" $3 "-" lang_id "_openslr " src_dir "/" $1 "/" $1 "_" $2 "_" $3 ".wav"}' \
                        ${data_folder}/line_index.tsv  > ${target_dir}/wav.scp
                fi
                sort ${target_dir}/utt2spk -o ${target_dir}/utt2spk
                sort ${target_dir}/text -o ${target_dir}/text
                sort ${target_dir}/wav.scp -o ${target_dir}/wav.scp
                utils/utt2spk_to_spk2utt.pl ${target_dir}/utt2spk > ${target_dir}/spk2utt
                echo "${data_folder}"
                utils/validate_data_dir.sh --non-print --no-feats ${target_dir}

                utils/subset_data_dir.sh \
                    data/openslr${openslr_id}_${lang_id} \
                    500 \
                    data/dev-test-${lang}-openslr${openslr_id}

                utils/subset_data_dir.sh \
                    data/dev-test-${lang}-openslr${openslr_id} \
                    250 \
                    data/dev_${lang}_openslr${openslr_id}

                utils/copy_data_dir.sh --validate_opts "--non-print" \
                    data/dev-test-${lang}-openslr${openslr_id} \
                    data/test_${lang}_openslr${openslr_id}

                utils/filter_scp.pl --exclude data/dev_${lang}_openslr${openslr_id}/wav.scp \
                    data/dev-test-${lang}-openslr${openslr_id}/wav.scp \
                    > data/test_${lang}_openslr${openslr_id}/wav.scp

                utils/fix_data_dir.sh data/test_${lang}_openslr${openslr_id}

                utils/copy_data_dir.sh --validate_opts "--non-print" \
                    data/openslr${openslr_id}_${lang_id} \
                    data/train_${lang}_openslr${openslr_id}

                utils/filter_scp.pl --exclude data/dev-test-${lang}-openslr${openslr_id}/wav.scp \
                    data/openslr${openslr_id}_${lang_id}/wav.scp \
                    > data/train_${lang}_openslr${openslr_id}/wav.scp

                utils/fix_data_dir.sh data/train_${lang}_openslr${openslr_id}
                test_set="${test_set} test_${lang}_openslr${openslr_id}"                
            done

            for dialect_lang in ${v1_processing_list}; do
                openslr_id=$(echo ${dialect_lang} | cut -f1 -d-)
                lang_id=$(echo ${dialect_lang} | cut -f2 -d-)
                gender=$(echo ${dialect_lang} | cut -f3 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}/${lang_id}_${gender}/
                target_dir=data/openslr${openslr_id}_${lang_id}_${gender}
                mkdir -p ${target_dir}

                awk -v lang_id="${lang_id}" -F '[_\t]' \
                    '{print $1 "_" $2 "_" $3 "-" lang_id "_openslr " "_" $1 "_" $2}' \
                    ${data_folder}/line_index.tsv  > ${target_dir}/utt2spk
                awk -v lang_id="${lang_id}" -F '[\t]' \
                    '{print $1 "-" lang_id "_openslr " $2}' \
                    ${data_folder}/line_index.tsv  > ${target_dir}/text
                awk -v lang_id="${lang_id}" -v src_dir="${data_folder}" -F '[_\t]' \
                    '{print $1 "_" $2 "_" $3 "-" lang_id "_openslr " src_dir "/wavs/" $1 "_" $2 "_" $3 ".wav"}' \
                    ${data_folder}/line_index.tsv  > ${target_dir}/wav.scp
                sort ${target_dir}/utt2spk -o ${target_dir}/utt2spk
                sort ${target_dir}/text -o ${target_dir}/text
                sort ${target_dir}/wav.scp -o ${target_dir}/wav.scp
                utils/utt2spk_to_spk2utt.pl ${target_dir}/utt2spk > ${target_dir}/spk2utt
                utils/validate_data_dir.sh --non-print --no-feats ${target_dir}

                utils/subset_data_dir.sh \
                    data/openslr${openslr_id}_${lang_id}_${gender} \
                    500 \
                    data/dev-test-${lang}-openslr${openslr_id}_${gender}

                utils/subset_data_dir.sh \
                    data/dev-test-${lang}-openslr${openslr_id}_${gender} \
                    250 \
                    data/dev_${lang}_openslr${openslr_id}_${gender}

                utils/copy_data_dir.sh --validate_opts "--non-print" \
                    data/dev-test-${lang}-openslr${openslr_id}_${gender} \
                    data/test_${lang}_openslr${openslr_id}_${gender}

                utils/filter_scp.pl --exclude data/dev_${lang}_openslr${openslr_id}_${gender}/wav.scp \
                    data/dev-test-${lang}-openslr${openslr_id}_${gender}/wav.scp \
                    > data/test_${lang}_openslr${openslr_id}_${gender}/wav.scp

                utils/fix_data_dir.sh data/test_${lang}_openslr${openslr_id}_${gender}

                utils/copy_data_dir.sh --validate_opts "--non-print" \
                    data/openslr${openslr_id}_${lang_id}_${gender} \
                    data/train_${lang}_openslr${openslr_id}_${gender}

                utils/filter_scp.pl --exclude data/dev-test-${lang}-openslr${openslr_id}_${gender}/wav.scp \
                    data/openslr${openslr_id}_${lang_id}_${gender}/wav.scp \
                    > data/train_${lang}_openslr${openslr_id}_${gender}/wav.scp

                utils/fix_data_dir.sh data/train_${lang}_openslr${openslr_id}_${gender}
                test_set="${test_set} test_${lang}_openslr${openslr_id}_${gender}"
            done 
        fi

    fi

    if [[ "${googlei18n_lang_asr}" == *"${lang}"* ]]; then

        case ${lang} in
            "jv")
                data_info="35-javanese" ;;
            "su")
                data_info="36-sundanese" ;;
            "si")
                data_info="52-sinhala" ;;
            "bn")
                data_info="53-bengali" ;;
            "ne")
                data_info="54-nepali" ;;
        esac

        openslr_id=$(echo ${data_info} | cut -f1 -d-)
        lang_id=$(echo ${data_info} | cut -f2 -d-)

        if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
            log "sub-stage 0: Download Data to ${GOOGLEI18N}"

            echo "skip download for openslr${openslr_id}_${lang_id}" 
            # idxs=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "a" "b" "c" "d" "e" "f")
            # for i in "${idxs[@]}"; do
                # wget -O ${GOOGLEi18N}/openslr${openslr_id}_${lang_id} \
                #     https://www.openslr.org/resources/${openslr_id}/asr_${lang_id}_${i}.zip
                # unzip -o asr_javanese_${i}.zip
            # done
            
        fi

        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            log "sub-stage 1: Data Preparation for Openslr ASR corpora"
            prefix_info=${GOOGLEI18N}/openslr${openslr_id}_${lang_id}/asr_${lang_id}/
            target_dir=data/openslr${openslr_id}_${lang_id}
            mkdir -p ${target_dir}

            awk -v lang_id="${lang_id}" -F '[\t]' \
                '{print $2 "_" $1 "-" lang_id "_openslr " $2}' \
                ${prefix_info}/utt_spk_text.tsv  > ${target_dir}/utt2spk

            awk -v lang_id="${lang_id}" -F '[_\t]' \
                '{print $2 "_" $1 "-" lang_id "_openslr " $3}' \
                ${prefix_info}/utt_spk_text.tsv  > ${target_dir}/text

            awk -v lang_id="${lang_id}" -v src_dir="${prefix_info}" -F '[_\t]' \
                '{print $2 "_" $1 "-" lang_id "_openslr " src_dir "data/" substr($1, 1, 2) "/" $1 ".flac"}' \
                ${prefix_info}/utt_spk_text.tsv  > ${target_dir}/wav.scp

            sort ${target_dir}/utt2spk -o ${target_dir}/utt2spk
            sort ${target_dir}/text -o ${target_dir}/text
            sort ${target_dir}/wav.scp -o ${target_dir}/wav.scp
            utils/utt2spk_to_spk2utt.pl ${target_dir}/utt2spk > ${target_dir}/spk2utt
            utils/validate_data_dir.sh --non-print --no-feats ${target_dir}
 
            utils/subset_data_dir.sh \
                data/openslr${openslr_id}_${lang_id} \
                4000 \
                data/dev-test-${lang}-openslr${openslr_id}

            utils/subset_data_dir.sh \
                data/dev-test-${lang}-openslr${openslr_id} \
                2000 \
                data/dev_${lang}_openslr${openslr_id}

            utils/copy_data_dir.sh --validate_opts "--non-print" \
                data/dev-test-${lang}-openslr${openslr_id} \
                data/test_${lang}_openslr${openslr_id}

            utils/filter_scp.pl --exclude data/dev_${lang}_openslr${openslr_id}/wav.scp \
                data/dev-test-${lang}-openslr${openslr_id}/wav.scp \
                > data/test_${lang}_openslr${openslr_id}/wav.scp

            utils/fix_data_dir.sh data/test_${lang}_openslr${openslr_id}

            utils/copy_data_dir.sh --validate_opts "--non-print" \
                data/openslr${openslr_id}_${lang_id} \
                data/train_${lang}_openslr${openslr_id}

            utils/filter_scp.pl --exclude data/dev-test-${lang}-openslr${openslr_id}/wav.scp \
                data/openslr${openslr_id}_${lang_id}/wav.scp \
                > data/train_${lang}_openslr${openslr_id}/wav.scp

            utils/fix_data_dir.sh data/train_${lang}_openslr${openslr_id}
            test_set="${test_set} test_${lang}_openslr${openslr_id}"  

        fi
    fi

    if [[ "${googlei18n_lang_30_32_37}" == *"${lang}"* ]]; then
        case ${lang} in
            "af")
                processing_list="32-af_za/za/afr" ;;
            "st")
                processing_list="32-st_za/za/sso" ;;
            "tn")
                processing_list="32-tn_za/za/tsn" ;;
            "xh")
                processing_list="32-xh_za/za/xho" ;;
            "bn")
                processing_list="37-bn_bd 37-bn_in" ;;
        esac

        if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
            log "sub-stage0: Download data to ${GOOGLEI18N}"
            echo "pending"
        fi

        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            log "sub-stage 1: Data Preparation for Openslr low-resource ASR corpora"
            for direct_lang in ${processing_list}; do
                openslr_id=$(echo ${direct_lang} | cut -f1 -d-)
                prefix=$(echo ${direct_lang} | cut -f2 -d-)
                data_folder=${GOOGLEI18N}/openslr${openslr_id}_${prefix}
                target_dir=data/openslr${openslr_id}_${lang}
                mkdir -p ${target_dir}

                awk -v lang="${lang}" -F '[_\t]' \
                    '{print $1 "_" $2 "_" $3 "-" lang_id "_openslr " $1 "_" $2}' \
                    ${data_folder}/line_index.tsv  > ${target_dir}/utt2spk

                awk -v lang="${lang}" -F '[\t]' \
                    '{print $1 "-" lang_id "_openslr " $2}' \
                    ${data_folder}/line_index.tsv  > ${target_dir}/text

                awk -v lang="${lang}" -v src_dir="${data_folder}" -F '[_\t]' \
                    '{print $1 "_" $2 "_" $3 "-" lang_id "_openslr " src_dir "/wavs/" $1 "_" $2 "_" $3 ".wav"}' \
                    ${data_folder}/line_index.tsv  > ${target_dir}/wav.scp

                sort ${target_dir}/utt2spk -o ${target_dir}/utt2spk
                sort ${target_dir}/text -o ${target_dir}/text
                sort ${target_dir}/wav.scp -o ${target_dir}/wav.scp
                utils/utt2spk_to_spk2utt.pl ${target_dir}/utt2spk > ${target_dir}/spk2utt
                utils/validate_data_dir.sh --non-print --no-feats ${target_dir}

                utils/subset_data_dir.sh \
                    data/openslr${openslr_id}_${lang} \
                    500 \
                    data/dev-test-${lang}-openslr${openslr_id}

                utils/subset_data_dir.sh \
                    data/dev-test-${lang}-openslr${openslr_id} \
                    250 \
                    data/dev_${lang}_openslr${openslr_id}

                utils/copy_data_dir.sh --validate_opts "--non-print" \
                    data/dev-test-${lang}-openslr${openslr_id} \
                    data/test_${lang}_openslr${openslr_id}

                utils/filter_scp.pl --exclude data/dev_${lang}_openslr${openslr_id}/wav.scp \
                    data/dev-test-${lang}-openslr${openslr_id}/wav.scp \
                    > data/test_${lang}_openslr${openslr_id}/wav.scp
                utils/fix_data_dir.sh data/test_${lang}_openslr${openslr_id}

                utils/copy_data_dir.sh --validate_opts "--non-print" \
                    data/openslr${openslr_id}_${lang} data/train_${lang}_openslr${openslr_id}

                utils/filter_scp.pl --exclude data/dev-test-${lang}-openslr${openslr_id}/wav.scp \
                    data/openslr${openslr_id}_${lang}/wav.scp \
                    > data/train_${lang}_openslr${openslr_id}/wav.scp

                utils/fix_data_dir.sh data/train_${lang}_openslr${openslr_id}
                test_set="${test_set} test_${lang}_openslr${openslr_id}"      
            done
        fi
    fi

    if [[ "${voxpopuli_lang}" == *"${lang}"* ]]; then
        if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
            log "sub-stage0: Download data to ${VOXPOPULI}"
        fi

        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            log "sub-stage 1: Data Preparation for Voxpopuli"
            for subset in "train" "test" "dev"; do
                index_file=${VOXPOPULI}/transcribed_data/${lang}/asr_${subset}.tsv
                target_dir=data/${subset}_${lang}_voxpopuli
                mkdir -p ${target_dir}
                # remove header
                awk '(NR>1)' ${index_file} > ${target_dir}/${subset}.no_header
                src_index=${target_dir}/${subset}.no_header

                awk -v lang="${lang}" -F '[\t]' \
                    '{printf "%010d_%s-%s_voxpopuli %s\n", $4, $1, lang, $3}' \
                    ${src_index} > ${target_dir}/text

                awk -v lang="${lang}" -F '[\t]' \
                    '{printf "%010d_%s-%s_voxpopuli %010d\n", $4, $1,lang, $4}'  \
                    ${src_index} > ${target_dir}/utt2spk
                awk -v lang="${lang}" -v src_dir="${VOXPOPULI}/transcribed_data/${lang}/" \
                    -F '[\t]' \
                   '{
                        printf "%010d_%s-%s_voxpopuli sox %s%s/%s.ogg -t wav - |\n",
                        $4, $1, lang, src_dir, substr($1, 1, 4), $1
                   }' \
                   ${src_index} > ${target_dir}/wav.scp
                sort ${target_dir}/utt2spk -o ${target_dir}/utt2spk
                sort ${target_dir}/text -o ${target_dir}/text
                sort ${target_dir}/wav.scp -o ${target_dir}/wav.scp
                utils/utt2spk_to_spk2utt.pl ${target_dir}/utt2spk > ${target_dir}/spk2utt
                utils/validate_data_dir.sh --no-feats ${target_dir}
            done
            test_set="${test_set} test_${lang}_voxpopuli"
        fi

    fi

done

# Processing extra langs
for lang in ${extra_langs}; do
        if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
            log "sub-stage 0: Download Data to ${COMMONVOICE}"

            # base url for downloads.
            data_url=https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-9.0-2022-04-27/cv-corpus-9.0-2022-04-27-${lang}.tar.gz

            local/voxforge/download_and_untar.sh ${COMMONVOICE} ${data_url} ${lang}.tar.gz
            # (Optional) remove archieve files
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
            utils/copy_data_dir.sh --validate_opts "--non-print" \
                data/"$(echo "validated_${lang}_commonvoice" | tr - _)" data/${train_subset}
            utils/filter_scp.pl --exclude data/${train_subdev}/wav.scp \
                data/${train_subset}/wav.scp > data/${train_subset}/temp_wav.scp
            utils/filter_scp.pl --exclude data/${test_subset}/wav.scp \
                data/${train_subset}/temp_wav.scp > data/${train_subset}/wav.scp
            utils/fix_data_dir.sh data/${train_subset}
        fi
        test_set="${test_set} ${test_subset}"
done

# Full test set for reference
'''
test_set="test_ab_commonvoice test_fy_NL_commonvoice test_or_commonvoice \
test_af_openslr32 test_ga_IE_commonvoice test_pa_IN_commonvoice \
test_ar_commonvoice test_gl_commonvoice test_pl \
test_as_commonvoice test_gl_openslr77 test_pl_commonvoice \
test_az_commonvoice test_gn_commonvoice test_pl_mls \
test_ba_commonvoice test_gu_openslr78 test_pl_voxpopuli \
test_bas_commonvoice test_ha_commonvoice test_pt_commonvoice \
test_be_commonvoice test_hi_commonvoice test_pt_mls \
test_bg_commonvoice test_hr_voxpopuli test_pt_voxforge \
test_bn_commonvoice test_hsb_commonvoice test_rm_sursilv_commonvoice \
test_bn_openslr37 test_hu_commonvoice \
test_bn_openslr53 test_hu_voxpopuli test_rm_vallader_commonvoice \
test_br_commonvoice test_hy_AM_commonvoice \
test_ca_commonvoice test_ia_commonvoice test_ro_commonvoice \
test_ca_openslr69 test_id_commonvoice test_ro_voxpopuli \
test_ckb_commonvoice test_ig_commonvoice test_ru_commonvoice \
test_cnh_commonvoice test_it_commonvoice test_ru_voxforge \
test_cs_commonvoice test_it_mls test_rw_commonvoice \
test_cs_voxpopuli test_it_voxforge test_sah_commonvoice \
test_cv_commonvoice test_it_voxpopuli test_sat_commonvoice \
test_cy_commonvoice test_ja_commonvoice test_si_openslr52 \
test_da_commonvoice test_jv_openslr35 test_sk_commonvoice \
test_de_commonvoice test_jv_openslr41_female test_sk_voxpopuli \
test_de_mls test_jv_openslr41_male test_sl_commonvoice \
test_de_voxforge test_kab_commonvoice test_sl_voxpopuli \
test_de_voxpopuli test_ka_commonvoice test_sr_commonvoice \
test_dv_commonvoice test_kk_commonvoice test_st_openslr32 \
test_el_commonvoice test_km_openslr42_male test_su_openslr36 \
test_en_commonvoice test_kmr_commonvoice test_su_openslr44_female \
test_en_mls test_su_openslr44_male \
test_en_openslr70 test_kn_openslr79 test_sv_SE_commonvoice \
test_en_voxforge test_ky_commonvoice \
test_en_voxpopuli test_lg_commonvoice test_sw_commonvoice \
test_eo_commonvoice test_lt_commonvoice test_ta_commonvoice \
test_es_commonvoice test_lt_voxpopuli test_ta_openslr65 \
test_es_mls test_lv_commonvoice test_te_openslr66 \
test_es_openslr61 test_mdf_commonvoice test_th_commonvoice \
test_es_openslr71 test_mhr_commonvoice test_tig_commonvoice \
test_es_openslr72 test_tn_openslr32 \
test_es_openslr73 test_mk_commonvoice test_tok_commonvoice \
test_es_openslr74 test_ml_commonvoice test_tr_commonvoice \
test_es_openslr75 test_ml_openslr63 test_tt_commonvoice \
test_es_voxforge test_mn_commonvoice test_ug_commonvoice \
test_es_voxpopuli test_mr_commonvoice test_uk_commonvoice \
test_et_commonvoice test_mr_openslr64 test_ur_commonvoice \
test_et_voxpopuli test_mt_commonvoice test_uz_commonvoice \
test_eu_commonvoice test_myv_commonvoice test_vi_commonvoice \
test_eu_openslr76 test_nan_tw_commonvoice test_vot_commonvoice \
test_fa_commonvoice test_ne_openslr43_female test_xh_openslr32 \
test_fi_commonvoice test_ne_openslr54 test_yo_openslr86 \
test_fi_voxpopuli test_nl_commonvoice test_yue_commonvoice \
test_fr_commonvoice test_nl_mls test_zh_CN_commonvoice \
test_fr_mls test_nl_voxforge test_zh_HK_commonvoice \
test_fr_voxforge test_nl_voxpopuli test_zh_TW_commonvoice \
test_fr_voxpopuli test_nn_NO_commonvoice"
'''

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
                | python3 -c 'import string; import sys; print(sys.stdin.read().translate(str.maketrans("", "", string.punctuation)), end="")') \
              > ${x}/text
        rm ${x}/text.org
        paste -d " " \
            <(cut -f 1 -d" " ${x}/utt2spk) \
            <(cut -f 1 -d" " ${x}/utt2spk) \
            > ${x}/utt2spk.identical
        mv ${x}/utt2spk.identical ${x}/utt2spk
    done

    for x in ${test_set}; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " \
              <(cut -f 1 -d" " data/${x}/text.org) \
              <(cut -f 2- -d" " data/${x}/text.org \
                | python3 -c 'import sys; print(sys.stdin.read().upper(), end="")' \
                | python3 -c 'import string; import sys; print(sys.stdin.read().translate(str.maketrans("", "", string.punctuation)), end="")') \
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
