#! /usr/bin/env bash 

# Copyright 2020 Ruhr-University (Wentao Yu)

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

# hand over parameters 
musan_root=$1			# Musan root directory
dset=$2				# Dataset part (Train, Test, Val, pretrain)
datatype=$3			# LRS2 or LRS3 dataset

# general configuration
stage=0
srcdir=data/audio/clean/$datatype		# source directory of clean audio files
augdatadir=data/audio/augment   # save directory for augmented audio files

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 0 ]; then
    utils/data/get_utt2dur.sh $srcdir/$dset
    utils/data/get_utt2num_frames.sh $srcdir/$dset
    frame_shift=0.01
    awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' $srcdir/$dset/utt2num_frames > $srcdir/$dset/reco2dur
    
    if [ ! -d "RIRS_NOISES" ]; then
    	# Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    	wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    	unzip rirs_noises.zip
    	rm -rf rirs_noises.zip
    fi
  
    # Make a version with reverberated speech
    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
    
    # Prepare the MUSAN corpus, which consists of music, speech, and noise
    # suitable for augmentation.
    steps/data/make_musan.sh --sampling-rate 16000 $musan_root data/audio
    len=`wc -l $srcdir/$dset/text | cut -d " " -f 1`
    subn=$(($len / 9))
    revbsubn=$(($len / 7))
    # Get the duration of the MUSAN recordings.  This will be used by the
    # script augment_data_dir.py.
    nameambient=noise
    namemusic=music
    name_list="${nameambient} ${namemusic}"

    
    
    for name in ${name_list};do
	utils/data/get_utt2dur.sh data/audio/musan_${name}
    	mv data/audio/musan_${name}/utt2dur data/audio/musan_${name}/reco2dur

    	# Augment with musan_noise
    	steps/data/augment_data_dir.py \
		--utt-suffix "noise" \
		--fg-interval 1 \
		--fg-snrs "12:9:6:3:0:-3:-6:-9:-12" \
		--fg-noise-dir data/audio/musan_${name} $srcdir/$dset $augdatadir/${datatype}_decode/${dset}_${name}

   	# Combine reverb, noise, music, and babble into one directory.
    	utils/subset_data_dir.sh $srcdir/$dset $subn $srcdir/${dset}$subn
    	oldstr='|'
    	newstr='; rm /tmp/tmp.$$ |'
    	sed -i "s%${oldstr}%${newstr}%g" $srcdir/${dset}$subn/wav.scp
    	oldstr='- - |'
    	newstr='- - ; rm /tmp/tmp.$$ |'
	sed -i "s%${oldstr}%${newstr}%g" $augdatadir/${datatype}_decode/${dset}_${name}/wav.scp

   	utils/combine_data.sh $augdatadir/${datatype}_decode/${dset}_aug_${name} $augdatadir/${datatype}_${dset}_reverb$revbsubn $augdatadir/${datatype}_decode/${dset}_${name} $srcdir/${dset}$subn
    done
    
fi

