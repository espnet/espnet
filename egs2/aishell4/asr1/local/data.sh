#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

 . utils/parse_options.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}



#################################################################
#####             Downloading their git          ################
#################################################################


# Github AISHELL4 : https://github.com/felixfuyihui/AISHELL-4.git
FOLDER=AISHELL-4   # voir à quel endroit le mettre ! 
URL=https://github.com/felixfuyihui/AISHELL-4.git

if [ ! -d "$FOLDER" ] ; then
    git clone "$URL" "$FOLDER"
    log "git successfully downloaded"
fi

#pip install -r "$FOLDER"/requirements.txt   # ATTENTION JE CHANGE LEURS REQUIERMENTS POUR METTRE SENTENCE PIECE 0.1.94 AU LIEU DE 0.1.91 QUI FAIT TOUT BUGUER
# IDEM POUR TORCH JE MET 1.9, voir le vrai git pour les versions d'origine ! 
# rappel : attention, j'ai modifié leur Git  --> oui, il faut que je le fork


#################################################################
#####            Downloading data and producing lists      ##############
#################################################################



if false ; then

    for room_name in "train_L" "train_M" "train_S" 
    do 

        wget https://www.openslr.org/resources/111/$room_name.tar.gz -P ${AISHELL4}/  
        
        
        tar -xzvf ${AISHELL4}/"$room_name".tar.gz -C ${AISHELL4}/
        

        # after that untar step, you have one folder "$room_name" with two subfolders : 
        #   - wav : a list of .flac audio files, each audio file is a conference meeting of about 30 minutes 
        #   - TextGrid : a list of .TextGrid and .rttm files 

        # then you have to produce a list of the names of the files located in the "$room_name"/wav/ directory 
        # list should be like : 
        #/ocean/projects/cis210027p/shared/corpora/AISHELL4/train_L/wav/20200707_L_R001S01C01.flac
        #/ocean/projects/cis210027p/shared/corpora/AISHELL4/train_L/wav/20200709_L_R002S06C01.flac
        #/ocean/projects/cis210027p/shared/corpora/AISHELL4/train_L/wav/20200707_L_R001S04C01.flac
        # ...

        rm  ${AISHELL4}/$room_name/wav_list.txt
        FILES="$PWD/${AISHELL4}/$room_name/wav/*"
        for f in $FILES
        do
            echo "$f" >> ${AISHELL4}/$room_name/wav_list.txt
        done



        # then you have to produce a list of the names of the .TextGrid files located in the "$room_name"/textgrid/ directory 
        # list should be like : 
        #/ocean/projects/cis210027p/shared/corpora/AISHELL4/train_L/TextGrid/textgrid_list/20200706_L_R001S08C01.TextGrid
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

if false; then 
    mkdir ${AISHELL4}/full_train
    rm ${AISHELL4}/full_train/wav_list.txt
    rm ${AISHELL4}/full_train/TextGrid_list.txt

    for r in "train_L" "train_M" "train_S" 
    do 
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

if true ; then 

    log "generating asr training data ..."
    log "(this can take some time)"
    rm -rf "$output_folder"
 
    python "$FOLDER"/data_preparation/generate_asr_trainingdata.py  --output_dir "$output_folder" --mode train --aishell4_wav_list "$wav_list_aishell4" --textgrid_list "$text_grid_aishell4" || log "ca a pas marché" ;

    log "asr training data generated."

fi
 



#################################################################
#####     creating wav.scp from output/train/wav directory    ##############
#################################################################


if true ; then 
    rm $output_folder/train/wav.scp
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

if true ; then 
    rm $output_folder/train/utt2spk
    FILES="$output_folder/train/wav/*"
    for f in $FILES
    do
        g=$(echo $f | cut -d'/' -f 14 | cut -d'.' -f 1) 
        echo "$g" "$g"  >> $output_folder/train/utt2spk  # we put speaker_id = utt_id
    done

    # creationg spk2utt from utt2spk
    rm $output_folder/train/spk2utt
    utils/utt2spk_to_spk2utt.pl $output_folder/train/utt2spk > $output_folder/train/spk2utt
fi 







#################################################################
#####            sort and fix the data  ########
#################################################################


if true ; then
    log "sorting files ... "
    sort data/train/utt2spk -o data/train/utt2spk
    sort data/train/wav.scp -o data/train/wav.scp
    sort data/train/text -o data/train/text
    log "files sorted"

    # then, removing empty lines

    log "fixing files ..."
    ./utils/fix_data_dir.sh data/train
    log "files fixed"
fi





#################################################################
#####            test                                    ########
#################################################################




if true ; then

    #wget https://www.openslr.org/resources/111/test.tar.gz -P ${AISHELL4}/  
    
    
    #tar -xzvf ${AISHELL4}/test.tar.gz -C ${AISHELL4}/
    

    rm  ${AISHELL4}/test/wav_list.txt
    FILES="$PWD/${AISHELL4}/test/wav/*"
    for f in $FILES
    do
        echo "$f" >> ${AISHELL4}/test/wav_list.txt
    done


    rm ${AISHELL4}/test/TextGrid_list.txt
    FILES="$PWD/${AISHELL4}/test/TextGrid/*.TextGrid"
    for f in $FILES
    do
        echo "$f" >> ${AISHELL4}/test/TextGrid_list.txt
    done

fi




#################################################################
#####            ground truth for asr, using aishell4 github     ##############
#################################################################


wav_list_aishell4=${AISHELL4}/test/wav_list.txt
text_grid_aishell4=${AISHELL4}/test/TextGrid_list.txt

output_folder=$PWD/data/test/

if true ; then 

    log "generating asr training data ..."
    log "(this can take some time)"
    rm -rf "$output_folder"
 
    python "$FOLDER"/data_preparation/generate_asr_trainingdata.py  --output_dir "$output_folder" --mode train --aishell4_wav_list "$wav_list_aishell4" --textgrid_list "$text_grid_aishell4" || log "ca a pas marché" ;

    log "asr training data generated."

    mv $output_folder/train/* $output_folder/
    rm -r $output_folder/train

fi
 



#################################################################
#####     creating wav.scp from output/train/wav directory    ##############
#################################################################


if true ; then 
    rm $output_folder/wav.scp
    FILES="$output_folder/wav/*"
    for f in $FILES
    do

        g=$(echo $f | cut -d'/' -f 14 | cut -d'.' -f 1) 
        echo "$g" "$f" >> $output_folder/wav.scp
    done

fi




#################################################################
#####            creating utt2spk and spk2utt  ########
#################################################################

if true ; then 
    rm $output_folder/utt2spk
    FILES="$output_folder/wav/*"
    for f in $FILES
    do
        g=$(echo $f | cut -d'/' -f 14 | cut -d'.' -f 1) 
        echo "$g" "$g"  >> $output_folder/utt2spk  # we put speaker_id = utt_id
    done

    # creationg spk2utt from utt2spk
    rm $output_folder/spk2utt
    utils/utt2spk_to_spk2utt.pl $output_folder/utt2spk > $output_folder/spk2utt
fi 




#################################################################
#####            sort and fix the data  ########
#################################################################


if true ; then
    log "sorting files ... "
    sort data/test/utt2spk -o data/test/utt2spk
    sort data/test/wav.scp -o data/test/wav.scp
    sort data/test/text -o data/test/text
    log "files sorted"

    # then, removing empty lines , IL FAUDRA TROUVER PQ JAI DES EMPTY LINES CEST CHELOU

    log "fixing files ..."
    ./utils/fix_data_dir.sh data/test
    log "files fixed"
fi

