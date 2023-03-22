#!/usr/bin/env bash

# Copyright  2020-2021  University of Stuttgart (Author: Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
nj=32

# feature configuration
dumpdir=dump   # directory to dump full features
do_delta=false

# decoding parameter
# here we use models trained with the LibriSpeech Transformer recipe
# https://github.com/espnet/espnet/blob/master/egs/librispeech/asr1/RESULTS.md#pytorch-large-transformer-with-specaug-4-gpus--transformer-lm-4-gpus
# they are downloaded by the local/download_asr.sh script
asr_url="https://drive.google.com/open?id=1RHYAhcnlKz08amATrf0ZOWFLzoQphtoc"
recog_model=model.val5.avg.best  # set a model to be used for decoding
lang_model=rnnlm.model.best # set a language model to be used for decoding

# directories for pre-trained models' downloads
#
# you can change them to directories with your own models,
# don't forget to disable the download in such case
#
# xvector directory needs to contain following files:
# - final.raw (x-vector extractor model)
# - extract.config (x-vector extractor configuration)
xvector_dir=download/xvector_voxceleb
# asr directory needs to contain following files:
# - $recog_model (ASR model, e.g. model.val5.avg.best) and model.json with its configuration
# - $lang_model (Language Model, e.g. rnnlm.model.best) and model.json with its configuration
# - cmvn.ark (CMVN statistics file)
# - *_units.txt (dictionary file, e.g. train_960_unigram5000_units.txt)
# - *.model (SentencePiece model, e.g. train_960_unigram5000.model)
# - decode.yaml (decoding script configuration)
asr_dir=download/asr_librispeech

diarizer_type=spectral # choose between spectral, bhmm or agglomerative

# If the following is set to true, we use the oracle speaker and segment
# information instead of performing SAD and diarization.
use_oracle_segments=false

# please change the path accordingly
libricss_corpus=/resources/asr-data/LibriCSS

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

recog_set="dev eval"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Models Download"
    local/data_download.sh ${libricss_corpus}
    local/download_xvector.sh ${xvector_dir}
    local/download_asr.sh ${asr_url} ${asr_dir}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    local/data_prep_mono.sh ${libricss_corpus}

    if [ ${use_oracle_segments} = true ]; then
        for rtask in ${recog_set}; do
            local/segment_diarize_oracle.sh data/${rtask} data/${rtask}_oracle
        done
    else
        xvector_model_dir=$(dirname "$(find ${xvector_dir} -name final.raw | head -n 1)")
        for rtask in ${recog_set}; do
            local/segment.sh data/${rtask}
            local/diarize.sh --cmd "$train_cmd" --nj ${nj} --diarizer-type ${diarizer_type} \
                ${xvector_model_dir} data/${rtask} \
                exp/diarize_${diarizer_type}/${rtask} \
                data/${rtask}_diarized_${diarizer_type}
        done
    fi
fi

tmp_recog_set=""

if [ ${use_oracle_segments} = true ]; then
    for rtask in ${recog_set}; do
        tmp_recog_set="${tmp_recog_set} ${rtask}_oracle"
    done
else
    for rtask in ${recog_set}; do
        tmp_recog_set="${tmp_recog_set} ${rtask}_diarized_${diarizer_type}"
    done
fi

recog_set=${tmp_recog_set}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    # dump features for training
    cmvn=$(find ${asr_dir} -name cmvn.ark | head -n 1)
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp ${cmvn} exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=$(find ${asr_dir} -name '*_units.txt' | head -n 1)
bpemodel=$(find ${asr_dir} -name '*.model' | head -n 1)
dict_affix=$(basename ${bpemodel} .model)
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Json Data Preparation"
    wc -l ${dict}

    # make json labels
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel} \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${dict_affix}.json
    done
fi

lmexpdir=$(dirname "$(find ${asr_dir} -name ${lang_model} | head -n 1)")

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    echo "Skipped, using pre-trained LM from ${lmexpdir}"
fi

asrexpdir=$(dirname "$(dirname "$(find ${asr_dir} -name ${recog_model} | head -n 1)")")

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    echo "Skipped, using pre-trained ASR from ${asrexpdir}"
fi

expdir=exp/$(basename ${asrexpdir})
mkdir -p ${expdir}
decode_config=$(find ${asr_dir} -name decode.yaml | head -n 1)

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    for rtask in ${recog_set}; do
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${dict_affix}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${dict_affix}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${asrexpdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/${lang_model} \
            --api v2 \
            --beam-size 30

        # next command is just for json to text conversion
        score_sclite.sh --bpe 1 --bpemodel ${bpemodel} --wer true ${expdir}/${decode_dir} ${dict} >/dev/null

        if [ ${use_oracle_segments} = true ]; then
            local/score_reco_oracle.sh ${expdir}/${decode_dir}
        else
            local/score_reco_diarized.sh ${expdir}/${decode_dir} data/${rtask}
        fi
    done
    echo "Finished"
fi
