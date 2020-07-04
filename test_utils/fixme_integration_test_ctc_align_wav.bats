#!/usr/bin/env bats
# -*- mode:sh -*-

setup() {
    tmpdir=/tmp/espnet-test-ctc-align
}


@test "ctc_align_wav" {
    cd ./egs/wsj/asr1/
    wav=../../../test_utils/ctc_align_test.wav
    transcription="THE SALE OF THE HOTELS IS PART OF HOLIDAY'S STRATEGY TO SELL OFF ASSETS AND CONCENTRATE ON PROPERTY MANAGEMENT"
    model=wsj.transformer.v1
    ../../../utils/ctc_align_wav.sh --stop-stage 2 --align_dir ${tmpdir} --models ${model} ${wav} "${transcription}"

    prefix="Alignment: "

    # NOTE(karita): If you will change the model, you should change these outputs.
    alignment=$(../../../utils/ctc_align_wav.sh --stage 3 --align_dir ${tmpdir} --models ${model} ${wav} "${transcription}" | grep "${prefix}")
    # Transform alignment to text for comparison.
    text=$(echo ${alignment} | awk '{output=$1" "; for (i=2; i<NF; i++) {if ($i!=$(i+1)) {output=output$i;}}; print output}' | sed -e 's/<blank>//g' -e 's/<space>/ /g')
    [ "${text}" = "${prefix}${transcription}" ]

}
