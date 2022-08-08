#!/usr/bin/env bash

# Copyright 2020 Ruhr-University (Wentao Yu)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


# general configuration
ifpretrain=true			# if use LRS2 pretrain set 
iflrs3pretrain=true		# if use LRS3 pretrain set
ifsegment=true  		# if do segmentation for pretrain set
ifcuda=true  			# if use cuda
ifmulticore=true        	# if multi cpu processing, default is true in all scripts
num=  			# this variable is related with next variable. Only applies when ifdebug=true
ifdebug=false	   		# with debug, we only use $num Utts from pretrain and $num Utts from Train set
backend=pytorch
stage=-1			# start from -1 if you need to start from data download
stop_stage=100			# stage at which to stop
dataprocessingstage=0		# stage for data processing in stage 3
stop_dataprocessingstage=100	# stage at which to stop
ngpu=1       			# number of gpus ("0" uses cpu, otherwise use gpu)
nj=16
debugmode=1
dumpdir=dump   			# directory to dump full features
N=0            			# number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      			# verbose option
train_lm=false			# true: Train own language model, false: use pretrained librispeech LM model

# Setting path variables for dataset, OpenFace, DeepXi, pretrained model and musan
# Change this variables and adapt it to your Folder structure
DATA_DIR=					# The LRS2 dataset directory e.g. "/home/foo/LRS2"
DATALRS3_DIR=				# The LRS3 dataset directory e.g. "/home/foo/LRS3"
PRETRAINEDMODEL=pretrainedvideomodel/Video_only_model.pt 				        # Path to pretrained video model e.g. "pretrainedvideomodel/Video_only_model.pt"
MUSAN_DIR="musan"   					              	#  The noise dataset directory e.g. "musan" 

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml 
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# bpemode (unigram or bpe)
nbpe=500
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

## Function for pretrained Librispeech language model:  
function gdrive_download () {
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
        "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
    rm -rf /tmp/cookies.txt
}

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# define sets
if [ "$ifpretrain" = true ] ; then
	train_set="pretrain_Train"
else
	train_set="Train"
fi
train_dev="Val"
recog_set="Val Test"




# Stage -1: download local folder 
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # download required files for data processing
    local/download.sh
fi

# Stage 0: install software
OPENFACE_DIR=local/installations/OpenFace/build/bin	# Path to OpenFace build directory
VIDAUG_DIR=local/installations/vidaug 		 	# Path to vidaug directory
DEEPXI_DIR=local/installations/DeepXi 			# DeepXi directory
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Install required softwares
    local/installpackage.sh $OPENFACE_DIR $VIDAUG_DIR $DEEPXI_DIR
fi

# Stage 1: Data preparation
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Data preparation"

    echo "Download pretrained video feature extractor and check directory configuration"
    if [ -f "$PRETRAINEDMODEL" ] ; then
	echo "pretrained video feature extractor already exists"
    else
        gdrive_download '1vwsJlUYgRUSDGDEa9RXail0mp4tR5QAx' 'model.v2.tar.gz'  || exit 1;
        tar -xf model.v2.tar.gz  || exit 1;
        mv model.v2/avsrlrs2_3/pretrainedvideomodel ./
        rm -rf model.v2
        rm -rf model.v2.tar.gz
    fi

    if [ -d "$DATA_DIR" ] ; then
	echo "Dataset already exists."
    else
	echo "For downloading the data, please visit 'https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html'."
	echo "You will need to sign a Data Sharing agreement with BBC Research & Development before getting access."
	echo "Please download the dataset by yourself and save the dataset directory in path.sh file"
	echo "Thanks!"
    fi
	
    if [ "$iflrs3pretrain" = true ] ; then
    	if [ -d "$DATALRS3_DIR" ]; then
    	    echo "Dataset already exists."
    	else
    	    echo "For downloading the data, please visit 'https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html'."
    	    echo "You will need to sign a Data Sharing agreement with BBC Research & Development before getting access."
    	    echo "Please download the dataset by yourself and save the dataset directory in path.sh file"
    	    echo "Thanks!"
    	fi
    fi

    # Create Musan directory
    if [ -d "${MUSAN_DIR}" ]; then
	echo "MUSAN dataset is in ${MUSAN_DIR}..."
    else
	echo "Download MUSAN dataset"
	wget --no-check-certificate http://www.openslr.org/resources/17/musan.tar.gz
	echo "Download finished"
	echo "Unzip MUSAN dataset"
	tar -xf musan.tar.gz
 	rm -rf musan.tar.gz
	echo "Unzipping finished"
    fi   
    # Create RIRS_NOISES Dataset
    if [ -d "RIRS_NOISES" ]; then
	echo "RIRS_NOISES dataset is in RIRS_NOISES..."
    else
    	# Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
	echo "Download RIRS_NOISES dataset"
    	wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
	echo "Download finished"
	echo "Unzip RIRS_NOISES dataset"
    	unzip rirs_noises.zip
    	rm -rf rirs_noises.zip
	echo "Unzipping finished"
    fi

    for part in Test Val Train; do 
        # use underscore-separated names in data directories. #Problem: Filelist_Val is readonly
        local/data_prepare/lrs2_audio_data_prep.sh ${DATA_DIR} $part $ifsegment $ifmulticore $ifdebug $num $nj || exit 1;
    done
    if [ "$ifpretrain" = true ] ; then
    	part=pretrain
    	local/data_prepare/lrs2_audio_data_prep.sh ${DATA_DIR} $part $ifsegment $ifmulticore $ifdebug $num $nj || exit 1;
    fi

    if [ "$iflrs3pretrain" = true ] ; then

	## embedding LRS3 code
    	python3 -m venv --system-site-packages ./LRS3-env
    	source ./LRS3-env/bin/activate
    	pip3 install pydub
    	local/data_prepare/lrs3_audio_data_prep.sh $DATALRS3_DIR pretrain $ifmulticore $ifsegment $ifdebug $num
    	deactivate
    	rm -rf ./LRS3-env
        mkdir -p data/audio/clean/LRS3/pretrain
	mv Dataset_processing/LRS3/kaldi/pretrainsegment/* data/audio/clean/LRS3/pretrain
	cp Dataset_processing/LRS3/audio/pretrain/Filelist_pretrain Dataset_processing/LRS3/audio/pretrain/Filelist_LRS3pretrain
	mv Dataset_processing/LRS3/audio/pretrain/Filelist_LRS3pretrain data/METADATA
    fi
    echo "stage 1: Data preparation finished"

fi

# Stage 2: Audio augmentation
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 2: Audio augmentation"
    for part in Test Val Train; do 
        # use underscore-separated names in data directories.
        local/extract_reliability/audio_augmentation.sh $MUSAN_DIR $part LRS2 || exit 1;
    done

    if [ "$ifpretrain" = true ] ; then
    	part=pretrain
    	local/extract_reliability/audio_augmentation.sh $MUSAN_DIR $part LRS2 || exit 1;
    fi
    if [ "$iflrs3pretrain" = true ] ; then
    	part=pretrain
    	local/extract_reliability/audio_augmentation.sh $MUSAN_DIR $part LRS3 || exit 1;
    fi
    # The Test set is augmented with ambient and music noise SNR from -12 to 12
    local/extract_reliability/audio_augmentation_recog.sh $MUSAN_DIR Test LRS2 || exit 1;
    echo "Datasets Combination"
    if [[ "$ifpretrain" = true || "$iflrs3pretrain" = true ]] ; then ## combine pretrain and train set
 	if [[ "$ifpretrain" = true && "$iflrs3pretrain" = false ]] ; then
		utils/combine_data.sh data/audio/augment/pretrain_Train_aug \
					data/audio/augment/LRS2_Train_aug \
					data/audio/augment/LRS2_pretrain_aug || exit 1;
		utils/combine_data.sh data/audio/augment/pretrain_aug \
					data/audio/augment/LRS2_pretrain_aug || exit 1;
	elif [[ "$ifpretrain" = false && "$iflrs3pretrain" = true ]] ; then
		utils/combine_data.sh data/audio/augment/pretrain_Train_aug \
					data/audio/augment/LRS2_Train_aug \
					data/audio/augment/LRS3_pretrain_aug || exit 1;
		utils/combine_data.sh data/audio/augment/pretrain_aug \
					data/audio/augment/LRS3_pretrain_aug || exit 1;
	elif [[ "$ifpretrain" = true && "$iflrs3pretrain" = true ]] ; then
		utils/combine_data.sh data/audio/augment/pretrain_Train_aug \
					data/audio/augment/LRS2_Train_aug \
					data/audio/augment/LRS2_pretrain_aug \
					data/audio/augment/LRS3_pretrain_aug  || exit 1;
		utils/combine_data.sh data/audio/augment/pretrain_aug \
					data/audio/augment/LRS2_pretrain_aug \
					data/audio/augment/LRS3_pretrain_aug || exit 1;
	fi
    fi
    mv data/audio/augment/LRS2_Test_aug data/audio/augment/Test_aug
    mv data/audio/augment/LRS2_Val_aug data/audio/augment/Val_aug
    mv data/audio/augment/LRS2_Train_aug data/audio/augment/Train_aug	

    echo "stage 2: Audio augmentation finished"

fi

mp3files=Dataset_processing/Audioaugments
feat_tr_dir=${dumpdir}/audio_org/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
# Stage 3: Feature Generation for audio features
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Feature Generation"
    echo "stage 3.1: Make augmented mp3 files"
    mkdir -p $mp3files
    if [ "$ifpretrain" = false ] && [ "$iflrs3pretrain" = false ] ; then
        for part in Test Val Train; do
	    echo "Run audioaugwav frames for ${part} set!" 
 	    mkdir -p ${mp3files}/$part
            local/extract_reliability/audioaugwav.sh data/audio/augment/${part}_aug $mp3files/$part || exit 1;
        done

    else
	for part in Test Val Train pretrain; do #Train pretrain 
	    echo "Run audioaugwav frames for ${part} set!" 
	    mkdir -p $mp3files/$part
    	    local/extract_reliability/audioaugwav.sh data/audio/augment/${part}_aug $mp3files/$part || exit 1;
        done

	part=pretrain
        python3 local/extract_reliability/segaugaudio.py $mp3files data/audio/augment $part $ifmulticore
	rm -r ${mp3files:?}/${part:?}
	mv ${mp3files}/${part}_aug $mp3files/${part}
    fi
    nameambient=noise
    namemusic=music
    name_list="${nameambient} ${namemusic}"
    for name in ${name_list};do
        dset=Test
     	mkdir -p ${mp3files}/${dset}_${name}  || exit 1;
	local/extract_reliability/audioaugwav.sh data/audio/augment/LRS2_decode/${dset}_aug_${name} $mp3files/${dset}_${name} || exit 1;
    done
    echo "stage 3.1: Make augmented mp3 files finished"

    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 3.2: Feature Generation"

    fbankdir=fbank
    mfccdir=mfccs
    if [[ "$ifpretrain" = true || "$iflrs3pretrain" = true ]] ; then ## combine pretrain and train set
	# Generate the fbank and mfcc features; by default 80-dimensional fbanks with pitch on each frame

	mv data/audio/augment/pretrain_aug/segments data/audio/augment/pretrain_aug/segments_old
	mv data/audio/augment/pretrain_Train_aug/segments data/audio/augment/pretrain_Train_aug/segments_old
	for x in pretrain Train Test Val; do #pretrain_Train pretrain Train
	    mv data/audio/augment/${x}_aug/wav.scp data/audio/augment/${x}_aug/wavnew.scp
	    python3 local/extract_reliability/remakewav.py data/audio/augment/${x}_aug/wavnew.scp data/audio/augment/${x}_aug/wav.scp Dataset_processing/Audioaugments/$x
	    cp -R data/audio/augment/${x}_aug data/audio/augment/${x}mfccs_aug
	    mv data/audio/augment/${x}_aug data/audio/augment/${x}fbank_aug
	    steps/make_mfcc.sh \
		--cmd "$train_cmd" \
		--nj $nj \
		--write_utt2num_frames true \
			data/audio/augment/${x}mfccs_aug \
			exp/make_mfcc/${x} \
			${mfccdir}  || exit 1;
            utils/fix_data_dir.sh data/audio/augment/${x}mfccs_aug  || exit 1;
	    steps/make_fbank_pitch.sh \
		--cmd "$train_cmd" \
		--nj $nj \
		--write_utt2num_frames true \
			data/audio/augment/${x}fbank_aug \
			exp/make_fbank/${x} \
			${fbankdir}  || exit 1;
            utils/fix_data_dir.sh data/audio/augment/${x}fbank_aug  || exit 1;
	done

	utils/combine_data.sh data/audio/augment/pretrain_Trainfbank_aug \
					data/audio/augment/pretrainfbank_aug \
					data/audio/augment/Trainfbank_aug  || exit 1;
	utils/combine_data.sh data/audio/augment/pretrain_Trainmfccs_aug \
					data/audio/augment/pretrainmfccs_aug \
					data/audio/augment/Trainmfccs_aug || exit 1;
    else
        # Generate the fbank and mfcc features; by default 80-dimensional fbanks with pitch on each frame
	for x in Train Val Test; do #
	    cp -R data/audio/augment/${x}_aug data/audio/augment/${x}mfccs_aug
	    mv data/audio/augment/${x}_aug data/audio/augment/${x}fbank_aug
	    steps/make_mfcc.sh \
		--cmd "$train_cmd" \
		--nj $nj \
		--write_utt2num_frames true \
			data/audio/augment/${x}mfccs_aug \
			exp/make_mfcc/${x} \
			${mfccdir}  || exit 1;
	    utils/fix_data_dir.sh data/audio/augment/${x}mfccs_aug  || exit 1;
	    steps/make_fbank_pitch.sh \
		--cmd "$train_cmd" \
		--nj $nj \
		--write_utt2num_frames true \
			data/audio/augment/${x}fbank_aug \
			exp/make_fbank/${x} \
			${fbankdir}  || exit 1;
	    utils/fix_data_dir.sh data/audio/augment/${x}fbank_aug  || exit 1;
	done
    fi 

    ## make fband and mfcc features for test decode dataset
    x=Test
    nameambient=noise
    namemusic=music
    name_list="${nameambient} ${namemusic}"
    for name in ${name_list};do
        rm -rf data/audio/augment/LRS2_decode/${x}mfccs_aug_${name}
        rm -rf data/audio/augment/LRS2_decode/${x}fbank_aug_${name}
	cp -R data/audio/augment/LRS2_decode/${x}_aug_${name} data/audio/augment/LRS2_decode/${x}mfccs_aug_${name}  || exit 1;
	mv data/audio/augment/LRS2_decode/${x}_aug_${name} data/audio/augment/LRS2_decode/${x}fbank_aug_${name}  || exit 1;
	steps/make_mfcc.sh \
		--cmd "$train_cmd" \
		--nj $nj \
		--write_utt2num_frames true \
		  data/audio/augment/LRS2_decode/${x}mfccs_aug_${name} \
		  exp/make_mfcc/${x}_${name} ${mfccdir}  || exit 1;
        utils/fix_data_dir.sh data/audio/augment/LRS2_decode/${x}mfccs_aug_${name}  || exit 1;
	steps/make_fbank_pitch.sh \
		--cmd "$train_cmd" \
		--nj $nj \
		--write_utt2num_frames true \
		  data/audio/augment/LRS2_decode/${x}fbank_aug_${name} \
		  exp/make_fbank/${x}_${name} ${fbankdir}  || exit 1;
        utils/fix_data_dir.sh data/audio/augment/LRS2_decode/${x}fbank_aug_${name}  || exit 1;
     done

    # compute global CMVN
    compute-cmvn-stats scp:data/audio/augment/${train_set}fbank_aug/feats.scp data/audio/augment/${train_set}fbank_aug/cmvn.ark  || exit 1;

    # dump features
    dump.sh  \
	--cmd "$train_cmd" \
	--nj $nj \
	--do_delta ${do_delta} \
	  data/audio/augment/${train_set}fbank_aug/feats.scp \
	  data/audio/augment/${train_set}fbank_aug/cmvn.ark \
	  exp/dump_feats/${train_set}fbank_aug ${feat_tr_dir}  || exit 1;

    for rtask in ${recog_set} Train pretrain; do
        feat_recog_dir=${dumpdir}/audio_org/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh \
	    --cmd "$train_cmd" \
            --nj $nj \
            --do_delta ${do_delta} data/audio/augment/${rtask}fbank_aug/feats.scp \
              data/audio/augment/${train_set}fbank_aug/cmvn.ark \
              exp/dump_feats/recog/${rtask} \
              ${feat_recog_dir}  || exit 1;
    done

    # make dump file for Test decode File
    nameambient=noise
    namemusic=music
    name_list="${nameambient} ${namemusic}"
    for name in ${name_list};do
        feat_recog_dir=${dumpdir}/audio_org/Test_decode_${name}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh \
	    --cmd "$train_cmd" \
            --nj $nj \
            --do_delta ${do_delta} data/audio/augment/LRS2_decode/Testfbank_aug_${name}/feats.scp \
              data/audio/augment/${train_set}fbank_aug/cmvn.ark \
              exp/dump_feats/recog/Test_${name} \
              ${feat_recog_dir}  || exit 1;
    done

    echo "stage 3.2: Audio Feature Generation finished"
    echo "stage 3: Feature Generation finished"
fi


dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
# Stage 4: Dictionary and JSON Data Preparation
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 4: Dictionary and Json Data Preparation"
    if [ "$train_lm" = true ] ; then
        mkdir -p data/lang_char/
     	echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    	cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
    	spm_train --input=data/lang_char/input.txt \
	    --vocab_size=${nbpe} \
	    --model_type=${bpemode} \
	    --model_prefix=${bpemodel} \
	    --input_sentence_size=100000000  || exit 1;
    	spm_encode --model=${bpemodel}.model \
	    --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}  || exit 1;
    	wc -l ${dict}
    else
    	# if using external librispeech lm        
	gdrive_download '1vwsJlUYgRUSDGDEa9RXail0mp4tR5QAx' 'model.v2.tar.gz'  || exit 1;
        tar -xf model.v2.tar.gz  || exit 1;
	mv model.v2/avsrlrs2_3/exp/train_rnnlm_pytorch_lm_unigram500 exp/train_rnnlm_pytorch_lm_unigram500
	mv model.v2/avsrlrs2_3/data/lang_char data/
	mv data/lang_char/train_unigram500.model data/lang_char/${train_set}_unigram500.model
    	mv data/lang_char/train_unigram500.vocab data/lang_char/${train_set}_unigram500.vocab
    	mv data/lang_char/train_unigram500_units.txt data/lang_char/${train_set}_unigram500_units.txt
	rm -rf model.v2
	rm -rf model.v2.tar.gz

    	##### it is depands on your corpus, if the corpus text transcription is uppercase, use this to convert to lowercase

    	textfilenames="data/audio/augment/*/text"
    	textdecodefilenames="data/audio/augment/LRS2_decode/*/text"
    	textcleanfilenames="data/audio/clean/*/*/text"
	for textname in $textfilenames $textdecodefilenames $textcleanfilenames; do
    	    for textfilename in $textname
    	    do
	    	sed -r 's/([^ \t]+\s)(.*)/\1\L\2/' $textfilename > ${textfilename}1  || exit 1;
	    	rm -rf $textfilename  || exit 1;
	    	mv ${textfilename}1 $textfilename  || exit 1;
    	    done
   	done
    fi

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
         data/audio/augment/${train_set}fbank_aug ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json  || exit 1;
    for rtask in ${recog_set} Train pretrain; do
	sed -r 's/([^ \t]+\s)(.*)/\1\L\2/' data/audio/augment/${rtask}fbank_aug/text > data/audio/augment/${rtask}fbank_aug/text1  || exit 1;
    	rm -rf data/audio/augment/${rtask}fbank_aug/text  || exit 1;
    	mv data/audio/augment/${rtask}fbank_aug/text1 data/audio/augment/${rtask}fbank_aug/text  || exit 1;
        
	feat_recog_dir=${dumpdir}/audio_org/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/audio/augment/${rtask}fbank_aug ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json  || exit 1;
    done

    ###make dump file for Test decode File
    nameambient=noise
    namemusic=music
    name_list="${nameambient} ${namemusic}"
    for name in ${name_list};do
	feat_recog_dir=${dumpdir}/audio_org/Test_decode_${name}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/audio/augment/LRS2_decode/Testfbank_aug_${name} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json  || exit 1;
    done

    echo "stage 4: Dictionary and Json Data Preparation finished"
fi


# Define new paths
facerecog=Dataset_processing/Facerecog
videoframe=Dataset_processing/Videodata
videoaug=Dataset_processing/Videoaug
videofeature=Dataset_processing/Videofeature
SNRdir=Dataset_processing/SNRsmat
SNRptdir=Dataset_processing/SNRs

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Extract reliability measures"
    if [ ${dataprocessingstage} -le 0 ] && [ ${stop_dataprocessingstage} -ge 0 ]; then
	#make mfcc dump file
	mkdir -p ${dumpdir}/mfcc/${train_set}/delta${do_delta}/  || exit 1;
        cp data/audio/augment/${train_set}mfccs_aug/feats.scp ${dumpdir}/mfcc/${train_set}/delta${do_delta}/  || exit 1;
    	data2json.sh --feat ${dumpdir}/mfcc/${train_set}/delta${do_delta}/feats.scp \
	   --bpecode ${bpemodel}.model data/audio/augment/${train_set}mfccs_aug ${dict} \
	   > ${dumpdir}/mfcc/${train_set}/delta${do_delta}/data_${bpemode}${nbpe}.json  || exit 1;
    	for rtask in ${recog_set} Train pretrain; do
            feat_recog_dir=${dumpdir}/mfcc/${rtask}/delta${do_delta}
     	    mkdir -p $feat_recog_dir  || exit 1;
	    cp data/audio/augment/${rtask}mfccs_aug/feats.scp ${dumpdir}/mfcc/${rtask}/delta${do_delta}/  || exit 1;
            data2json.sh --feat ${feat_recog_dir}/feats.scp \
	        --bpecode ${bpemodel}.model data/audio/augment/${rtask}mfccs_aug ${dict} \
		  > ${dumpdir}/mfcc/${rtask}/delta${do_delta}/data_${bpemode}${nbpe}.json  || exit 1;
        done

        nameambient=noise
        namemusic=music
        name_list="${nameambient} ${namemusic}"
        for name in ${name_list};do
    	    dset=Test
            feat_recog_dir=${dumpdir}/mfcc/Test_decode_${name}/delta${do_delta}
     	    mkdir -p $feat_recog_dir  || exit 1;
	    cp data/audio/augment/LRS2_decode/Testmfccs_aug_${name}/feats.scp ${dumpdir}/mfcc/Test_decode_${name}/delta${do_delta}/  || exit 1;
            data2json.sh --feat ${feat_recog_dir}/feats.scp \
	 	--bpecode ${bpemodel}.model data/audio/augment/LRS2_decode/Testmfccs_aug_${name} ${dict} \
		  > ${dumpdir}/mfcc/Test_decode_${name}/delta${do_delta}/data_${bpemode}${nbpe}.json  || exit 1;

            done
    fi

    if [ ${dataprocessingstage} -le 1 ] && [ ${stop_dataprocessingstage} -ge 1 ]; then
	#Stage 5.1: Video augmentation with Gaussian blur and salt&pepper noise
	if [ -d vidaug ]; then
  	    echo "vidaug already exist..."
	else
  	    ln -s $VIDAUG_DIR vidaug
	    ln -rsf local/extract_reliability/videoaug.py  vidaug/videoaug.py  
	fi
	python3 vidaug/videoaug.py data/METADATA/Filelist_Test $DATA_DIR $videoaug blur	# video augmentation with Gaussian blur
	python3 vidaug/videoaug.py data/METADATA/Filelist_Test $DATA_DIR $videoaug saltandpepper # video augmentation with salt and pepper noise
	unlink ./vidaug
    fi

    if [ ${dataprocessingstage} -le 2 ] && [ ${stop_dataprocessingstage} -ge 2 ]; then
	#Stage 5.2: Video stream processing, using OpenFace for face recognition
    	echo "stage 5.2: OpenFace face recognition"
	mkdir -p $facerecog
    	for part in Test Val Train; do  #
	    echo "Starting OpenFace background processes for ${part} set!"  
 	    mkdir -p $facerecog/LRS2${part}
            local/extract_reliability/Openface.sh $DATA_DIR $facerecog/LRS2${part} $part $OPENFACE_DIR \
				LRS2 $nj $ifdebug || exit 1;
    	done
    	if [ "$ifpretrain" = true ] ; then
    	    part=pretrain
    	    echo "Starting OpenFace background processes for ${part} set!"  
	    mkdir -p $facerecog/LRS2${part}
    	    local/extract_reliability/Openface.sh $DATA_DIR $facerecog/LRS2${part} $part $OPENFACE_DIR \
				LRS2 $nj $ifdebug || exit 1;
   	fi
    	if [ "$iflrs3pretrain" = true ] ; then
    	    part=pretrain
    	    echo "Starting OpenFace background processes for LRS3 ${part} set!"  
	    mkdir -p $facerecog/LRS3${part}
    	    local/extract_reliability/Openface.sh $DATALRS3_DIR $facerecog/LRS3${part} $part $OPENFACE_DIR \
				LRS3 $nj $ifdebug || exit 1;
   	fi
	part=Test
    	for noisetype in blur saltandpepper; do 
	    echo "Starting OpenFace background processes for ${part} set!"  
 	    mkdir -p $facerecog/LRS2${part}_$noisetype
            local/extract_reliability/Openface_vidaug.sh $videoaug $facerecog/LRS2${part}_$noisetype \
				$part $OPENFACE_DIR LRS2 $noisetype $nj $ifdebug || exit 1;
    	done

	echo "All OpenFace background processes for all sets are done!"
    fi

    if [ ${dataprocessingstage} -le 3 ] && [ ${stop_dataprocessingstage} -ge 3 ]; then	
	# Stage 5.3: Extract Video frames from the MP4 File by using OpenFace results
	echo "stage 5.3: Extract Frames"
	mkdir -p $videoframe

    	if [ "$ifpretrain" = true ] ; then
    	    part=pretrain
	    echo "Extracting frames for ${part} set!" 
	    mkdir -p $videoframe/LRS2${part}
    	    local/extract_reliability/extractframs.sh $DATA_DIR \
			$videoframe \
			$facerecog \
			data/audio/clean/LRS2 \
			$part \
			LRS2 \
			$ifsegment \
			$ifmulticore || exit 1;
   	fi

    	if [ "$iflrs3pretrain" = true ] ; then
    	    part=pretrain
	    echo "Extracting frames for ${part} set!" 
	    mkdir -p $videoframe/LRS3${part}
    	    local/extract_reliability/extractframs.sh $DATALRS3_DIR \
			$videoframe \
			$facerecog \
			data/audio/clean/LRS3 \
			$part \
			LRS3 \
			$ifsegment \
			$ifmulticore || exit 1;
   	fi

	for part in Test Val Train; do  # Test 
    	    echo "Extracting frames for ${part} set!"  	
 	    mkdir -p $videoframe/LRS2${part}
            local/extract_reliability/extractframs.sh $DATA_DIR \
			$videoframe \
			$facerecog \
			data/audio/clean/LRS2 \
			$part \
			LRS2 \
			$ifsegment \
			$ifmulticore || exit 1;
    	done
	part=Test
    	for noisetype in blur saltandpepper; do 
	    echo "Extracting frames for augumented ${part} set!" 
	    mkdir -p $videoframe/LRS2${part}_$noisetype
    	    local/extract_reliability/extractframs.sh $videoaug \
			$videoframe \
			$facerecog \
			data/audio/clean/LRS2 \
			$part \
			LRS2 \
			$ifsegment \
			$ifmulticore \
			$noisetype || exit 1;
	done
	echo "Extract Frames finished"	
    fi

    if [ ${dataprocessingstage} -le 4 ] && [ ${stop_dataprocessingstage} -ge 4 ]; then
        # Stage 5.4: Use DeepXi to estimate SNR
        echo "stage 5.4: Estimate SNRs using DeepXi framework"
	if [ -d DeepXi ]; then
  	    echo "Deepxi already exist..."
	else
  	    ln -s $DEEPXI_DIR DeepXi
	fi
	rm -rf DeepXi/set/test_noisy_speech
	rm -rf DeepXi/deepxi/se_batch.py
	cp local/se_batch.py DeepXi/deepxi
        if [ "$ifpretrain" = false ] && [ "$iflrs3pretrain" = false ] ; then
	    for part in Test Val Train; do
		echo "Extract SNR for ${part} set!"
 		mkdir -p $SNRdir/$part
                mkdir -p $SNRptdir/$part
        	local/extract_reliability/extractsnr.sh $SNRdir $SNRptdir $mp3files $part $ifmulticore || exit 1;
    	    done
    	else	
	    for part in Train pretrain Test Val; do  
		echo "Extract SNR for ${part} set!" 
		mkdir -p $SNRdir/$part
                mkdir -p $SNRptdir/$part
    		local/extract_reliability/extractsnr.sh $SNRdir $SNRptdir $mp3files $part $ifmulticore || exit 1;
	    done
   	fi
	nameambient=noise
        namemusic=music
        name_list="${nameambient} ${namemusic}"
        for name in ${name_list};do
    	    dset=Test
 	    mkdir -p $SNRdir/${dset}_${name}
     	    mkdir -p $SNRptdir/${dset}_${name}  || exit 1;
	    local/extract_reliability/extractsnr.sh $SNRdir $SNRptdir $mp3files ${dset}_${name} $ifmulticore || exit 1;
        done

	# Clean Up DeepXi: unlink and rm DeepXi
        unlink ./DeepXi
	rm -rf $SNRdir
    fi

    if [ ${dataprocessingstage} -le 5 ] && [ ${stop_dataprocessingstage} -ge 5 ]; then
	# Extract video features from video frames, if it is necessary
	echo "stage 5.5: Extract video features"
	mkdir -p $videofeature
	for part in Test Val; do
	    echo "Extract video features for ${part} set!"
	    mkdir -p $videofeature/LRS2${part}
	    local/extract_reliability/extractfeatures.sh $videoframe/LRS2${part}/Pics \
				$videofeature/LRS2${part} \
				$PRETRAINEDMODEL \
				$part \
				$ifcuda \
				$ifdebug || exit 1;
	done

    	if [ "$ifpretrain" = true ] ; then
	    part=pretrain
	    echo "Extract video features for ${part} set!"
	    mkdir -p $videofeature/LRS2${part}
	    local/extract_reliability/extractfeatures.sh $videoframe/LRS2${part}/Pics \
				$videofeature/LRS2${part} \
				$PRETRAINEDMODEL \
				$part \
				$ifcuda \
				$ifdebug || exit 1;
   	fi

    	if [ "$iflrs3pretrain" = true ] ; then
	    part=pretrain
	    echo "Extract video features for ${part} set!"
	    mkdir -p $videofeature/LRS3${part}
	    local/extract_reliability/extractfeatures.sh $videoframe/LRS3${part}/Pics \
				$videofeature/LRS3${part} \
				$PRETRAINEDMODEL \
				$part \
				$ifcuda \
				$ifdebug || exit 1;
   	fi
	part=Test
    	for noisetype in blur saltandpepper; do 
	    echo "Extract video features for augmented ${part} set!"
	    mkdir -p $videofeature/LRS2${part}_$noisetype
    	    local/extract_reliability/extractfeatures.sh $videoframe/LRS2${part}_$noisetype/Pics \
			$videofeature/LRS2${part}_$noisetype \
			$PRETRAINEDMODEL \
			$part \
			$ifcuda \
			$ifdebug || exit 1;
	done
    fi

    if [ ${dataprocessingstage} -le 6 ] && [ ${stop_dataprocessingstage} -ge 6 ]; then	
	# Make video ark files 
	echo "stage 5.6: Make video ark files"

	rm -rf data/video
	python3 local/extract_reliability/tensor2ark.py $videofeature data/video $nj
	for part in Test Val; do
	    echo "Make video dump files for LRS2 ${part} set!"
	    cat data/video/LRS2${part}/feats_*.scp > data/video/LRS2${part}/feats.scp || exit 1;
            sort data/video/LRS2${part}/feats.scp -o data/video/LRS2${part}/feats.scp
   	    mkdir -p ${dumpdir}/video/${part} || exit 1;
	    for files in text wav.scp utt2spk; do
		cp data/audio/clean/LRS2/${part}/${files} data/video/LRS2${part} || exit 1;
	    done
	    utils/fix_data_dir.sh data/video/LRS2${part}  || exit 1;
	    cp data/video/LRS2${part}/feats.scp ${dumpdir}/video/${part} || exit 1;
	    data2json.sh --feat ${dumpdir}/video/${part}/feats.scp --bpecode ${bpemodel}.model \
         			data/video/LRS2${part} ${dict} > ${dumpdir}/video/${part}/data_${bpemode}${nbpe}.json  || exit 1;
	done

	if [[ "$ifpretrain" = true || "$iflrs3pretrain" = true ]] ; then
	    part=pretrain
 	    if [[ "$ifpretrain" = true && "$iflrs3pretrain" = false ]] || [[ "$ifpretrain" = false && "$iflrs3pretrain" = true ]]; then
		if [[ "$ifpretrain" = true && "$iflrs3pretrain" = false ]] ; then
		    dataset=LRS2
		elif [[ "$ifpretrain" = false && "$iflrs3pretrain" = true ]] ; then
		    dataset=LRS3
		fi
		echo "Make video dump files for ${dataset} ${part} set!"
		mkdir -p data/video/${part}
	        cat data/video/${part}/feats_*.scp > data/video/${part}/feats.scp || exit 1;
		sort data/video/${part}/feats.scp -o data/video/${part}/feats.scp
   	        mkdir -p ${dumpdir}/video/${part} || exit 1;
		for files in text wav.scp utt2spk; do
		    cp data/audio/clean/${dataset}/${part}/${files} data/video/${part} || exit 1;
		done
		utils/fix_data_dir.sh data/video/${part}  || exit 1;
		cp data/video/${part}/${part}/feats.scp ${dumpdir}/video/${part} || exit 1;
	    elif [[ "$ifpretrain" = true && "$iflrs3pretrain" = true ]] ; then
		echo "Make video dump files for LRS2 and LRS3 ${part} set!"
	        cat data/video/LRS2${part}/feats_*.scp > data/video/LRS2${part}/feats.scp || exit 1;
	        cat data/video/LRS3${part}/feats_*.scp > data/video/LRS3${part}/feats.scp || exit 1;
		mkdir -p data/video/${part}
   		mkdir -p ${dumpdir}/video/${part} || exit 1;
		for files in text wav.scp utt2spk; do
		    cat data/audio/clean/LRS2/${part}/${files} data/audio/clean/LRS3/${part}/${files} > data/video/${part}/${files} || exit 1;
		    sort data/video/${part}/${files} -o data/video/${part}/${files}
  		done
		utils/fix_data_dir.sh data/video/${part}  || exit 1;
		cat data/video/LRS2${part}/feats.scp data/video/LRS3${part}/feats.scp > ${dumpdir}/video/${part}/feats.scp || exit 1;
		sort ${dumpdir}/video/${part}/feats.scp -o ${dumpdir}/video/${part}/feats.scp
	    fi

	    data2json.sh --feat ${dumpdir}/video/${part}/feats.scp --bpecode ${bpemodel}.model \
         			data/video/${part} ${dict} > ${dumpdir}/video/${part}/data_${bpemode}${nbpe}.json || exit 1;

	fi

	part=Test
    	for noisetype in blur saltandpepper; do 
	    echo "Make video dump files for augmented ${part} set!"
	    cat data/video/LRS2${part}_${noisetype}/feats_*.scp > data/video/LRS2${part}_${noisetype}/feats.scp || exit 1;
 	    sort data/video/LRS2${part}_${noisetype}/feats.scp -o data/video/LRS2${part}_${noisetype}/feats.scp
   	    mkdir -p ${dumpdir}/video/${part}_decode_${noisetype} || exit 1;
	    for files in text wav.scp utt2spk; do
		cp data/audio/clean/LRS2/${part}/${files} data/video/LRS2${part}_${noisetype} || exit 1;
	    done
	    utils/fix_data_dir.sh data/video/LRS2${part}_${noisetype}  || exit 1;
	    cp data/video/LRS2${part}_${noisetype}/feats.scp ${dumpdir}/video/${part}_decode_${noisetype} || exit 1;
	    data2json.sh --feat ${dumpdir}/video/${part}_decode_${noisetype}/feats.scp --bpecode ${bpemodel}.model \
         			data/video/LRS2${part}_${noisetype} ${dict} > ${dumpdir}/video/${part}_decode_${noisetype}/data_${bpemode}${nbpe}.json \
                                || exit 1;
	done
    fi

    if [ ${dataprocessingstage} -le 7 ] && [ ${stop_dataprocessingstage} -ge 7 ]; then
	# Remake dump files
	echo "stage 5.7: Remake audio and video dump files"

	for dset in pretrain_Train Val Test Test_decode_music Test_decode_noise; do 
            rm -rf dump/audio/$dset
	    python3 local/dump/audiodump.py dump/audio dump/audio_org $dset $ifmulticore || exit 1;
        done

	for dset in pretrain Val Test; do 
 	    rm -rf dump/avpretrain/$dset
	    python3 local/dump/avpretraindump.py dump/avpretrain dump/audio_org dump/video \
						$SNRptdir $videoframe dump/mfcc \
						$dset $ifmulticore || exit 1;
        done

	for dset in Train Val Test; do 
            rm -rf dump/avtrain/$dset
	    python3 local/dump/avtraindump.py dump/avtrain dump/audio_org $videofeature \
						$SNRptdir $videoframe dump/mfcc \
						$dset $ifmulticore || exit 1;
        done

	# Creat video dump file
	for dset in pretrain Val Test; do 
	    rm -rf dump/videopretrain/$dset
	    python3 local/dump/videodump.py dump/avpretrain dump/videopretrain $dset || exit 1;
        done

	for dset in Train Val Test; do 
	    rm -rf dump/videotrain/$dset
	    python3 local/dump/videodump.py dump/avtrain dump/videotrain $dset || exit 1;
        done

	dset=Test
	rm -rf dump/avpretraindecode
	rm -rf dump/avtraindecode
	for noisecombination in 'noise_None' 'music_None' 'noise_blur' 'noise_saltandpepper'; do 
	    python3 local/dump/avpretraindecodedump.py dump/avpretraindecode dump/audio_org dump/video \
				$SNRptdir $videoframe dump/mfcc \
				$dset $noisecombination $ifmulticore || exit 1;
	    python3 local/dump/avtraindecodedump.py dump/avtraindecode dump/audio_org dump/video \
				$videofeature $SNRptdir $videoframe dump/mfcc \
				$dset $noisecombination $ifmulticore || exit 1;
	done

    fi

    if [ ${dataprocessingstage} -le 8 ] && [ ${stop_dataprocessingstage} -ge 8 ]; then
	echo "stage 5.8: Split Test decode dump files"
	for audionoise in noise music; do
	    python3 local/extract_reliability/extractsnr.py data/audio/augment/LRS2_decode $audionoise $ifmulticore || exit 1;
        done
	for noisecombination in 'noise_None' 'music_None' 'noise_blur' 'noise_saltandpepper'; do 
	    python3 local/extract_reliability/splitsnr.py dump/avpretraindecode $noisecombination data/audio/augment/LRS2_decode || exit 1;
	    python3 local/extract_reliability/splitsnr.py dump/avtraindecode $noisecombination data/audio/augment/LRS2_decode || exit 1;
	done
    fi
	
    echo "stage 5: Reliability measures generation finished"
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 6)
# Otherwise, the pretrained Librispeech LM can be used (train_lm=false)

if [ "$train_lm" = false ] ; then
    lmexpname=train_rnnlm_pytorch_lm_unigram500
    lmexpdir=exp/${lmexpname}
else
    if [ -z ${lmtag} ]; then
        lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
    lmexpdir=exp/${lmexpname}
    mkdir -p ${lmexpdir}
fi

# Stage 6: Language Model (LM) preparation
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    if [ "$train_lm" = false ] ; then
        echo "stage 6: Use pretrained LM"
    else
        echo "stage 6: LM Preparation"
        lmdatadir=data/local/lm_train_${bpemode}${nbpe}
        # use external data
        if [ ! -e data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
	    echo "Download Librispeech normnalized language model (LM) training text"
            wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm_train/
	    echo "Download finished"
        fi
		
        if [ ! -e ${lmdatadir} ]; then
	    echo "Prepare LM data"
            mkdir -p ${lmdatadir}
	    # build gzip archive for language data out of the utterances in the LRS dataset
            cut -f 2- -d" " data/${train_set}/text | gzip -c > data/local/lm_train/${train_set}_text.gz
            # combine external text and transcriptions and shuffle them with seed 777
            zcat data/local/lm_train/librispeech-lm-norm.txt.gz data/local/lm_train/${train_set}_text.gz |\
                spm_encode \
                    --model=${bpemodel}.model \
                    --output_format=piece \
                > ${lmdatadir}/train.txt
            cut -f 2- -d" " data/audio/augment/${train_dev}fbank_aug/text | \
                spm_encode \
                    --model=${bpemodel}.model \
                    --output_format=piece \
                > ${lmdatadir}/valid.txt
	    echo "Preparation step done"
        fi
        echo "Start training Language Model"
        ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
            lm_train.py \
            --config ${lm_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --verbose 1 \
            --outdir ${lmexpdir} \
            --tensorboard-dir tensorboard/${lmexpname} \
            --train-label ${lmdatadir}/train.txt \
            --valid-label ${lmdatadir}/valid.txt \
            --resume ${lm_resume} \
            --dict ${dict} \
            --dump-hdf5-path ${lmdatadir}
        echo "stage 6: LM Preparation finished"
    fi
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend} #_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then 
	expname=${expname}_$(basename ${preprocess_config%.*}) 
    fi
else
    expname=${train_set}_${backend}_${tag}
fi

# ToDo: Hand over parameters for subscripts
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Stage 7: Network Training"
    # train audio model
    expdirapretrain=exp/pretrain/A
    mkdir -p ${expdirapretrain}
    echo ${expdirapretrain}
    noisetype=noise 	# Which noise type data is used for decoding, possible noisetype: noise music blur and saltandpepper
    local/training/train_audio.sh --backend $backend \
				--ngpu $ngpu \
				--debugmode $debugmode \
				--N $N \
 				--verbose $verbose \
				--nbpe $nbpe \
				--bpemode $bpemode \
				--nj $nj \
				--do_delta $do_delta \
			 	--train_set $train_set \
				--train_dev $train_dev \
				--preprocess_config $preprocess_config \
				--train_config $train_config\
				--lm_config $lm_config \
				--decode_config $decode_config\
				$expdirapretrain dump/audio dump/avpretraindecode $lmexpdir $noisetype $dict $bpemodel || exit 1;

    # pretrain video model
    expdirvpretrain=exp/pretrain/V
    mkdir -p ${expdirvpretrain}
    echo ${expdirvpretrain}
    noisetype=blur 	# Which noise type data is used for decoding, possible noisetype: noise music blur and saltandpepper
    local/training/pretrain_video.sh --backend $backend \
				--ngpu $ngpu \
				--debugmode $debugmode \
				--N $N \
 				--verbose $verbose \
				--nbpe $nbpe \
				--bpemode $bpemode \
				--nj $nj \
				--do_delta $do_delta \
				--preprocess_config $preprocess_config \
				--train_config $train_config\
				--lm_config $lm_config \
				--decode_config $decode_config\
				 $expdirvpretrain dump/videopretrain dump/avpretraindecode $lmexpdir $noisetype $dict $bpemodel  || exit 1;

    # finetune video model
    expdirvfine=exp/fine/V
    mkdir -p ${expdirvfine}
    echo ${expdirvfine}
    noisetype=blur 	# Which noise type data is used for decoding, possible noisetype: noise music blur and saltandpepper
    local/training/finetune_video.sh --backend $backend \
				--ngpu $ngpu \
				--debugmode $debugmode \
				--N $N \
 				--verbose $verbose \
				--nbpe $nbpe \
				--bpemode $bpemode \
				--nj $nj \
				--do_delta $do_delta \
				--preprocess_config $preprocess_config \
				--train_config $train_config\
				--lm_config $lm_config \
				--decode_config $decode_config\
				 $expdirvfine $expdirvpretrain dump/videotrain dump/avtraindecode $PRETRAINEDMODEL $lmexpdir $noisetype $dict $bpemodel  || exit 1;

    # pretrain audio-visual model
    expdiravpretrain=exp/pretrain/AV
    mkdir -p ${expdiravpretrain}
    echo ${expdiravpretrain}
    noisetype=noise 	# Which noise type data is used for decoding, possible noisetype: noise music blur and saltandpepper
    local/training/pretrain_av.sh --backend $backend \
				--ngpu $ngpu \
				--debugmode $debugmode \
				--N $N \
 				--verbose $verbose \
				--nbpe $nbpe \
				--bpemode $bpemode \
				--nj $nj \
				--do_delta $do_delta \
				--preprocess_config $preprocess_config \
				--train_config $train_config\
				--lm_config $lm_config \
				--decode_config $decode_config\
			        $expdiravpretrain dump/avpretrain dump/avpretraindecode $lmexpdir \
 				$noisetype $dict $bpemodel $expdirapretrain $expdirvpretrain|| exit 1;

    # finetune audio-visual model (final network used for decoding)
    expdiravfine=exp/fine/AV
    mkdir -p ${expdiravfine}
    echo ${expdiravfine}
    noisetype=noise 	# Which noise type data is used for decoding, possible noisetype: noise music blur and saltandpepper
    local/training/finetune_av.sh --backend $backend \
				--ngpu $ngpu \
				--debugmode $debugmode \
				--N $N \
 				--verbose $verbose \
				--nbpe $nbpe \
				--bpemode $bpemode \
				--nj $nj \
				--do_delta $do_delta \
				--preprocess_config $preprocess_config \
				--train_config $train_config\
				--lm_config $lm_config \
				--decode_config $decode_config\
				$expdiravfine dump/avtrain dump/avtraindecode $PRETRAINEDMODEL $lmexpdir \
 				$noisetype $dict $bpemodel $expdiravpretrain|| exit 1;

fi 

exit 0
