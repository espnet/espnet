#!/bin/bash

# Copyright 2020 Academia Sinica (Pin-Jui Ku)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpu in training
nj=8        # number of parallel jobs
dumpdir=dump # directory to dump full features
verbose=1    # verbose option (if set > 1, get more log)
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# face feature extraction related
fps=50
lip_width=128
lip_height=64
shape_predictor_path=downloads/shape_predictor_68_face_landmarks.dat

# config files
train_config=conf/train_pytorch_tacotron2_pretrained+spkemb.yaml
decode_config=conf/decode.yaml

# decoding related
model=snapshot.ep.300
n_average=0 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this
# if you're not on the CLSP grid.
db_root=downloads

# pretrained model related
CDVQVAE_model_config=conf/config_cdvqvae.json
CDVQVAE_model_path=${db_root}/CDVQVAE_1dcnn_num128_dim256.pt

# exp tag
tag="cdvqvae_codebook_128" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
dev_set=dev
eval_set=test

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/data_download.sh ${db_root}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    local/data_prep.sh ${db_root}/TMSV data/all
    # make train, dev and eval sets
    utils/subset_data_dir.sh --utt-list data/all/train_utt_list data/all data/${train_set}
    utils/fix_data_dir.sh data/${train_set}
    utils/subset_data_dir.sh --utt-list data/all/dev_utt_list data/all data/${dev_set}
    utils/fix_data_dir.sh data/${dev_set}
    utils/subset_data_dir.sh --utt-list data/all/eval_utt_list data/all data/${eval_set}
    utils/fix_data_dir.sh data/${eval_set}
    # the utils/subset_data_dir.sh do not split the video.scp file for us, so we need to do this seperately
    utils/filter_scp.pl data/${train_set}/utt2spk <data/all/video.scp >data/${train_set}/video.scp
    utils/filter_scp.pl data/${dev_set}/utt2spk <data/all/video.scp >data/${dev_set}/video.scp
    utils/filter_scp.pl data/${eval_set}/utt2spk <data/all/video.scp >data/${eval_set}/video.scp
fi

feat_tr_dir=${dumpdir}/${train_set}_fbank; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}_fbank; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}_fbank; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    fbankdir=fbank
    # make train, dev and eval sets
    utils/subset_data_dir.sh --utt-list data/all/train_utt_list data/all data/${train_set}
    utils/fix_data_dir.sh data/${train_set}
    utils/subset_data_dir.sh --utt-list data/all/dev_utt_list data/all data/${dev_set}
    utils/fix_data_dir.sh data/${dev_set}
    utils/subset_data_dir.sh --utt-list data/all/eval_utt_list data/all data/${eval_set}
    utils/fix_data_dir.sh data/${eval_set}

    for x in $train_set $dev_set $eval_set; do
        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${x} \
            exp/make_fbank/${x} \
            ${fbankdir}
    done

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    #remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set} data/${train_set}
    #remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${dev_set} data/${dev_set}

    # compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/eval ${feat_ev_dir}
    echo "fbank generation and normalization succeed."
fi

face_feat_tr_dir=${dumpdir}/${train_set}_face; mkdir -p ${face_feat_tr_dir}
face_feat_dt_dir=${dumpdir}/${dev_set}_face; mkdir -p ${face_feat_dt_dir}
face_feat_ev_dir=${dumpdir}/${eval_set}_face; mkdir -p ${face_feat_ev_dir}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Face Feature Generation"

    face_feature_dir=face_feature
    for x in ${dev_set} ${eval_set} ${train_set}; do

        make_face.sh --cmd "${train_cmd}" --nj ${nj} \
            --fps ${fps} \
            --lip_width ${lip_width} \
            --lip_height ${lip_height} \
            --shape_predictor_path ${shape_predictor_path} \
            data/${x} \
            exp/make_face/${x} \
            ${face_feature_dir}

    done
            
    # compute statistics for global mean-variance normalization for fbank
    compute-cmvn-stats scp:data/${train_set}/face_feats.scp data/${train_set}/face_cmvn.ark
    face_cmvn=data/${train_set}/face_cmvn.ark
    
    # dump features
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/face_feats.scp ${face_cmvn} exp/dump_face_feats/${train_set} ${face_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/face_feats.scp ${face_cmvn} exp/dump_face_feats/${dev_set} ${face_feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/face_feats.scp ${face_cmvn} exp/dump_face_feats/${eval_set} ${face_feat_ev_dir}
    echo "face-feature generation and normalization succeed."

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Use the pretrained lip VQVAE model to generate the text file"
    for name in $train_set $dev_set $eval_set; do
        python local/CDVQVAE/quantize_the_lip_image.py \
            -c ${CDVQVAE_model_config} \
            -p ${CDVQVAE_model_path} \
            -f ${dumpdir}/${name}_face/feats.scp \
            -o data/${name}/text
    done
fi

dict=data/lang_1char/${train_set}_units.txt
tr_dir=${dumpdir}/${train_set}_${tag}; mkdir -p ${tr_dir}
dt_dir=${dumpdir}/${dev_set}_${tag}; mkdir -p ${dt_dir}
ev_dir=${dumpdir}/${eval_set}_${tag}; mkdir -p ${ev_dir}
echo "dictionary: ${dict}"
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 4: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -t phn data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type phn \
         data/${train_set} ${dict} > ${tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type phn \
         data/${dev_set} ${dict} > ${dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp --trans_type phn \
         data/${eval_set} ${dict} > ${ev_dir}/data.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    for name in ${train_set} ${dev_set} ${eval_set}; do
        utils/copy_data_dir.sh data/${name} data/${name}_mfcc_16k
        utils/data/resample_data_dir.sh 16000 data/${name}_mfcc_16k
        steps/make_mfcc.sh \
            --write-utt2num-frames true \
            --mfcc-config conf/mfcc.conf \
            --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc_16k exp/make_mfcc_16k ${mfccdir}
        utils/fix_data_dir.sh data/${name}_mfcc_16k
        sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc_16k exp/make_vad ${vaddir}
        utils/fix_data_dir.sh data/${name}_mfcc_16k
    done

    # Check pretrained model existence
    nnet_dir=exp/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ]; then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    # Extract x-vector
    for name in ${train_set} ${dev_set} ${eval_set}; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
            ${nnet_dir} data/${name}_mfcc_16k \
            ${nnet_dir}/xvectors_${name}
    done
    # Update json
    for name in ${train_set} ${dev_set} ${eval_set}; do
        local/update_json.sh ${dumpdir}/${name}_${tag}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
    done
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Text-to-speech model training"
    tr_json=${tr_dir}/data.json
    dt_json=${dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

if [ ${n_average} -gt 0 ]; then
    model=model.last${n_average}.avg.best
fi
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding"
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}_${tag}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Synthesis"
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/fbank_cmvn.ark \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp
        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            --iters ${griffin_lim_iters} \
            ${outdir}_denorm/${name} \
            ${outdir}_denorm/${name}/log \
            ${outdir}_denorm/${name}/wav
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Synthesis useing ParallelWaveGAN"

    PWG_DIR=downloads/pwg
    PWG_steps=495000
    pids=() # initialize pids
    
    for name in ${dev_set} ${eval_set}; do
    (
        ls "${outdir}/${name}/feats.scp"
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/fbank_cmvn.ark \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp
        voc_expdir=${PWG_DIR}
        voc_checkpoint=${PWG_DIR}/checkpoint-${PWG_steps}steps.pkl

        # variable settings
        voc_conf="$(find "${voc_expdir}" -name "config.yml" -print0 | xargs -0 ls -t | head -n 1)"
        voc_stats="$(find "${voc_expdir}" -name "stats.h5" -print0 | xargs -0 ls -t | head -n 1)"
        wav_dir=${outdir}_denorm/${name}/pwg_wav
        hdf5_norm_dir=${outdir}_denorm/${name}/hdf5_norm
        [ ! -e "${wav_dir}" ] && mkdir -p ${wav_dir}
        [ ! -e ${hdf5_norm_dir} ] && mkdir -p ${hdf5_norm_dir}

        # normalize and dump them
        echo "Normalizing..."
        ${train_cmd} "${hdf5_norm_dir}/normalize.log" \
        parallel-wavegan-normalize \
            --skip-wav-copy \
            --config "${voc_conf}" \
            --stats "${voc_stats}" \
            --feats-scp "${outdir}_denorm/${name}/feats.scp" \
            --dumpdir ${hdf5_norm_dir} \
            --verbose "${verbose}"
        echo "successfully finished normalization."

        # decoding
        echo "Decoding start. See the progress via ${wav_dir}/decode.log."
        ${cuda_cmd} --gpu 1 "${wav_dir}/decode.log" \
            parallel-wavegan-decode \
                --dumpdir ${hdf5_norm_dir} \
                --checkpoint "${voc_checkpoint}" \
                --outdir ${wav_dir} \
                --verbose "${verbose}"
        echo "successfully finished decoding."
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    mcep_dim=24
    shift_ms=5
    num_of_spks=18
    for count_of_spk in $(seq 1 1 $num_of_spks); do
        spk=SP$(printf "%02d" $count_of_spk)

        out_wavdir=${outdir}_denorm/${eval_set}/pwg_wav
        gt_wavdir=${db_root}/TMSV/${spk}/audio
        minf0=$(awk '{print $1}' ${db_root}/TMSV/conf/${spk}.f0)
        echo "$minf0"
        maxf0=$(awk '{print $2}' ${db_root}/TMSV/conf/${spk}.f0)
        echo "$maxf0"
        out_spk_wavdir=${outdir}_denorm.ob_eval/mcd_eval/pwg_out/${spk}
        gt_spk_wavdir=${outdir}_denorm.ob_eval/mcd_eval/gt/${spk}
        mcd_file=${outdir}_denorm.ob_eval/mcd_eval/${spk}_pwg_mcd.log
        mkdir -p ${out_spk_wavdir}
        mkdir -p ${gt_spk_wavdir}
        
        local/make_spk_dir_for_mcd_eval.sh ${out_wavdir} ${gt_wavdir} \
            ${out_spk_wavdir} ${gt_spk_wavdir}
        ${decode_cmd} ${mcd_file} \
            mcd_calculate.py \
                --wavdir ${out_spk_wavdir} \
                --gtwavdir ${gt_spk_wavdir} \
                --mcep_dim ${mcep_dim} \
                --shiftms ${shift_ms} \
                --f0min ${minf0} \
                --f0max ${maxf0}
           
    done
fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "Stage 11: objective evaluation on ASR"
    local/ob_eval/evaluate.sh ${outdir} ${eval_set}
fi
