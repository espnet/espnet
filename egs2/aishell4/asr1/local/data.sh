#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
FOLDER=git_aishell

 . utils/parse_options.sh || exit 1;

mkdir -p ${AISHELL4}
if [ -z "${AISHELL4}" ]; then
    log "Fill the value of 'AISHELL4' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

#################################################################
#####             Downloading their git          ################
#################################################################


# Github AISHELL4 : https://github.com/felixfuyihui/AISHELL-4.git
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] ; then
    URL=https://github.com/DanBerrebbi/AISHELL-4.git
    # our fork

    if [ ! -d "$FOLDER" ] ; then
        git clone "$URL" "$FOLDER"
        log "git successfully downloaded"
    fi

    pip install -r "$FOLDER"/requirements.txt

fi



#################################################################
#####            Downloading data and producing lists      ##############
#################################################################



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] ; then

    for room_name in "train_L" "train_M" "train_S" "test"
    do

        wget https://www.openslr.org/resources/111/$room_name.tar.gz -P ${AISHELL4}/


        tar -xzvf ${AISHELL4}/"$room_name".tar.gz -C ${AISHELL4}/


        # after that untar step, you have one folder "$room_name" with two subfolders :
        #   - wav : a list of .flac audio files, each audio file is a conference meeting of about 30 minutes
        #   - TextGrid : a list of .TextGrid and .rttm files

        # then you have to produce a list of the names of the files located in the "$room_name"/wav/ directory
        # list should be like :
        #/dataset_dir/corpora/aishell4/train_L/wav/20200707_L_R001S01C01.flac
        #/dataset_dir/corpora/aishell4/train_L/wav/20200709_L_R002S06C01.flac
        #/dataset_dir/corpora/aishell4/train_L/wav/20200707_L_R001S04C01.flac
        # ...

        rm  ${AISHELL4}/$room_name/wav_list.txt
        FILES="$PWD/${AISHELL4}/$room_name/wav/*"
        for f in $FILES
        do
            echo "$f" >> ${AISHELL4}/$room_name/wav_list.txt
        done



        # then you have to produce a list of the names of the .TextGrid files located in the "$room_name"/textgrid/ directory
        # list should be like :
        #/dataset_dir/corpora/aishell4/train_L/TextGrid/textgrid_list/20200706_L_R001S08C01.TextGrid
        # ...

        rm ${AISHELL4}/$room_name/TextGrid_list.txt
        FILES="$PWD/${AISHELL4}/$room_name/TextGrid/*.TextGrid"
        for f in $FILES
        do
            echo "$f" >> ${AISHELL4}/$room_name/TextGrid_list.txt
        done

    done
fi


#################################################################
#####            Join train_L, train_M and train_S       ########
#################################################################

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    mkdir -p ${AISHELL4}/full_train
    for r in train_L train_M train_S ; do
        cat ${AISHELL4}/$r/TextGrid_list.txt >> ${AISHELL4}/full_train/TextGrid_list.txt
        cat ${AISHELL4}/$r/wav_list.txt >> ${AISHELL4}/full_train/wav_list.txt
    done
fi



#################################################################
#####            ground truth for asr, using aishell4 github     ##############
#################################################################


wav_list_aishell4=${AISHELL4}/full_train/wav_list.txt
text_grid_aishell4=${AISHELL4}/full_train/TextGrid_list.txt

output_folder=$PWD/data/

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] ; then

    log "generating asr training data ..."
    log "(this can take some time)"

    python "$FOLDER"/data_preparation/generate_asr_trainingdata.py  --output_dir "$output_folder" --mode train --aishell4_wav_list "$wav_list_aishell4" --textgrid_list "$text_grid_aishell4" || log "ca a pas marché" ;

    log "asr training data generated."

fi




#################################################################
#####     creating wav.scp from output/train/wav directory    ##############
#################################################################


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] ; then
    FILES="$output_folder/train/wav/*"
    for f in $FILES
    do
        g=$(echo $f | cut -d'/' -f 14 | cut -d'.' -f 1)
        echo "$g" "$f" >> $output_folder/train/wav.scp
    done

fi


#################################################################
#####            creating utt2spk and spk2utt  ########
#################################################################

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] ; then
    FILES="$output_folder/train/wav/*"
    for f in $FILES
    do
        g=$(echo $f | cut -d'/' -f 14 | cut -d'.' -f 1)
        echo "$g" "$g"  >> $output_folder/train/utt2spk  # we put speaker_id = utt_id
    done


fi







#################################################################
#####            sort and fix the data  ########
#################################################################


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ] ; then
    log "sorting files ... "
    sort data/train/utt2spk -o data/train/utt2spk
    # creating spk2utt from utt2spk
    utils/utt2spk_to_spk2utt.pl $output_folder/train/utt2spk > $output_folder/train/spk2utt
    sort data/train/wav.scp -o data/train/wav.scp
    sort data/train/text -o data/train/text
    log "files sorted"

    # then, removing empty lines

    log "fixing files ..."
    ./utils/fix_data_dir.sh data/train/
    log "files fixed"
fi


########################## generate the nlsyms.txt list


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ] ; then
    echo data/train/text | perl -pe 's/(\<[^\>\<]+\>)/$1\n/g' | perl -pe 's/(\<[^\>\<]+\>)/\n$1/' | grep "^\<.*\>$" | sort -u > data/nlsyms.txt
fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] ; then
    log "random shuffling to prepare dev and test sets ..."

    get_seeded_random()
        {
        seed="$1"
        openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
            </dev/zero 2>/dev/null
        }

    shuf  --random-source=<(get_seeded_random 76) data/train/utt2spk  -o data/train/utt2spk
    shuf  --random-source=<(get_seeded_random 76) data/train/wav.scp  -o data/train/wav.scp
    shuf  --random-source=<(get_seeded_random 76) data/train/text  -o data/train/text

fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ] ; then
    log "selecting lines for train, dev and test ..."

    utils/subset_data_dir.sh --first data/train 1000 data/dev
    n=$(($(wc -l < data/train/text) - 1000))
    utils/subset_data_dir.sh --last data/train ${n} data/train_nodev

fi


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ] ; then
    log "resorting the files ..."
    log "train ..."
    sort data/train_nodev/utt2spk -o data/train_nodev/utt2spk
    utils/utt2spk_to_spk2utt.pl data/train_nodev/utt2spk > data/train_nodev/spk2utt
    sort data/train_nodev/wav.scp -o data/train_nodev/wav.scp
    sort data/train_nodev/text -o data/train_nodev/text
    log "files sorted"
    log "dev ..."
    sort data/dev/utt2spk -o data/dev/utt2spk
    utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
    sort data/dev/wav.scp -o data/dev/wav.scp
    sort data/dev/text -o data/dev/text
    log "files sorted"


fi






#################################################################
#####      Combining with aishell1 data  (train only for now)
#################################################################


# pay attention : sorting issues with utt2spk :  (fix this by making speaker-ids prefixes of utt-ids)

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ] ; then

    aishell1_data=../../aishell/asr1/data/train
    aishell4_data=data/train_nodev

    u2s=$aishell1_data/utt2spk
    awk 'BEGIN {FS=" "; OFS="\n"}; {print $1" "$1}' $u2s > $aishell1_data/utt2spk2

    mv $aishell1_data/utt2spk2 $aishell1_data/utt2spk

    utils/combine_data.sh data/combined_aishell_dir/train $aishell1_data $aishell4_data


    sort data/combined_aishell_dir/train/utt2spk -o data/combined_aishell_dir/train/utt2spk
    utils/utt2spk_to_spk2utt.pl data/combined_aishell_dir/train/utt2spk > data/combined_aishell_dir/train/spk2utt
    sort data/combined_aishell_dir/train/wav.scp -o data/combined_aishell_dir/train/wav.scp
    sort data/combined_aishell_dir/train/text -o data/combined_aishell_dir/train/text

    wc -l data/combined_aishell_dir/train/*


fi





##########################
##       test set
##########################


wav_list_aishell4=${AISHELL4}/test/wav_list.txt
text_grid_aishell4=${AISHELL4}/test/TextGrid_list.txt

output_folder=$PWD/data

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ] ; then

    log "generating asr training data ..."
    log "(this can take some time)"

    python "$FOLDER"/data_preparation/generate_asr_trainingdata.py  --output_dir "$output_folder"/test --mode train --aishell4_wav_list "$wav_list_aishell4" --textgrid_list "$text_grid_aishell4" || log "ca a pas marché" ;

    log "asr training data generated."

    mv data/test/train/* data/test/
    rm -r data/test/train

fi



if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ] ; then
    FILES="$output_folder/test/wav/*"
    for f in $FILES
    do
        g=$(echo $f | cut -d'/' -f 13 | cut -d'.' -f 1)
        echo "$g" "$f" >> $output_folder/test/wav.scp
    done

fi


if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ] ; then
    FILES="$output_folder/test/wav/*"
    for f in $FILES
    do
        g=$(echo $f | cut -d'/' -f 13 | cut -d'.' -f 1)
        echo "$g" "$g"  >> $output_folder/test/utt2spk  # we put speaker_id = utt_id
    done


fi


if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ] ; then
    log "sorting files ... "
    sort data/test/utt2spk -o data/test/utt2spk
    # creating spk2utt from utt2spk
    utils/utt2spk_to_spk2utt.pl $output_folder/test/utt2spk > $output_folder/test/spk2utt
    sort data/test/wav.scp -o data/test/wav.scp
    sort data/test/text -o data/test/text
    log "files sorted"

    # then, removing empty lines

    log "fixing files ..."
    ./utils/fix_data_dir.sh data/test/
    log "files fixed"
fi
