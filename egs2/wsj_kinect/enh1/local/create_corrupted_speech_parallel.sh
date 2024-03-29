stage=0
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Should contain 2speakers directory
wsj_data_path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/ssivasankaran/experiments/data/speech_separation/wsj0-mix/"
chime5_wav_base='/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/CHiME5/audio/'
dihard_sad_label_path="$HOME/experiments/data/dihard/LDC2019E31_Second_DIHARD_Challenge_Development_Data/data/multichannel/sad/"
dest="dataset/2speakers_reverb_kinect"

. ./parse_options.sh

mkdir -p $dest

# Path to download the simulated RIRs. Dont change
rir_download_path="https://zenodo.org/record/3520737/files/kinect_rir.tar.gz"
min_max="max"
src_count=2 # Not implemented for other src counts
tt_parallel="007"
tr_parallel="049"
cv_parallel="012"
tr_SNR=10
cv_SNR=10
tt_SNR=15
chime_noise_dest="$dest/chime5_noise"

if [ $stage -le 0 ]; then
    # Download the RIRs
    echo "Stage 0: Download the RIR"
    if [ ! -d $dest/kinect_rir ]; then
        wget $rir_download_path -O $dest/kinect_rir.tar.gz
        tar -xvzf $dest/kinect_rir.tar.gz -C $dest || exit 1;
        # Remove the downloaded tar file to save space
        rm $dest/kinect_rir.tar.gz
        ln -s $dest/kinect_rir
    else
        echo "$dest/kinect_rir already exists. Skipping download"
    fi
fi

if [ $stage -le 1 ]; then
    # Extract chime noise
    echo "Stage 1: Extract chime noise"
    bash noise_from_chime5/getNonSpeechSegments.sh $chime5_wav_base $chime_noise_dest  $dihard_sad_label_path || exit 1;
fi

if [ $stage -le 2 ]; then
    echo "Stage 2: Create corrupted speech"
    pids=() # initialize pids
    for  dataset in tr tt cv; do
        if [ $dataset == 'tr' ]; then
            parallel_max=$tr_parallel
            SNR=$tr_SNR
        elif [ $dataset == 'tt' ]; then
            parallel_max=$tt_parallel
            SNR=$tt_SNR
        elif [ $dataset == 'cv' ]; then
            parallel_max=$cv_parallel
            SNR=$cv_SNR
        fi
        wsj_mix_list="list/wsj0-${src_count}mix_${dataset}.flist"
        #path to wsj-2mix dataset
        wsj2_mix_base="$wsj_data_path/${src_count}speakers/wav16k/${min_max}/${dataset}"
        noise_list="$chime_noise_dest/lists/$dataset"
        dest_base="$dest/${src_count}speakers_reverb_kinect_chime_noise_corrected/wav16k/min/${dataset}/"
        mkdir -p $dest_base
        rir_base_path="kinect_rir/${dataset}/"
        start=0
        for ele in `seq -w 000 $parallel_max`; do
            rir_yaml_list="${rir_base_path}/${ele}/list.yaml"
            rir_file_cnt=`grep rir_base_path $rir_yaml_list  | wc -l`
            file_end=$(($start+${rir_file_cnt}))
            echo $start $file_end
            # Put an & at the end of next line if you want run the wav creation in parallel using a single machine. It will spawn 50, 13 and 7 jobs for train dev and test
            # You can also use cluster managers such as SLURM/SGE/OAR with this command
            # python create_corrupted_speech.py $wsj_mix_list $start $file_end $wsj2_mix_base $noise_list $src_count $rir_yaml_list $SNR $dest_base  || exit 1;
            python create_corrupted_speech.py $wsj_mix_list $start $file_end $wsj2_mix_base $noise_list $src_count $rir_yaml_list $SNR $dest_base  || exit 1 &
            start=$file_end
        done
        # Remove wait if you can run 50+8+14 jobs  in parallel
        #wait
	pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
wait
