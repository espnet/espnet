#!/usr/bin/env bats
# -*- mode:sh -*-

setup() {
    tmpdir=/tmp/espnet-test-ctc-align
}


@test "ctc_align_wav" {
    cd ./egs/tedlium2/align1/
    mkdir -p conf
    cp ../../wsj/asr1/conf/no_preprocess.yaml ./conf
    wav=../../../test_utils/ctc_align_test.wav
    base=ctc_align_test
    transcription="THE SALE OF THE HOTELS IS PART OF HOLIDAY'S STRATEGY TO SELL OFF ASSETS AND CONCENTRATE ON PROPERTY MANAGEMENT"
    mkdir -p ${tmpdir}
    echo "batchsize: 0" > ${tmpdir}/align.yaml

    model=wsj.transformer_small.v1
    # if the model uses subsampling, adapt this factor accordingly
    subsampling_factor=1
    echo "${base} THE SALE OF THE HOTELS" > ${tmpdir}/utt_text
    echo "${base} IS PART OF HOLIDAY'S STRATEGY" >> ${tmpdir}/utt_text
    echo "${base} TO SELL OFF ASSETS" >> ${tmpdir}/utt_text
    echo "${base} AND CONCENTRATE" >> ${tmpdir}/utt_text
    echo "${base} ON PROPERTY MANAGEMENT" >> ${tmpdir}/utt_text


    ../../../utils/asr_align_wav.sh \
        --python "coverage run --append" \
        --models ${model} \
        --verbose 2  \
        --align_dir ${tmpdir} \
        --subsampling_factor ${subsampling_factor} \
        --min-window-size 100 \
        --align_config ${tmpdir}/align.yaml \
        ${wav} ${tmpdir}/utt_text
}
