#!/usr/bin/env bats
# -*- mode:sh -*-

setup() {
    cd tools || exit 1
    make moses.done
    cd - || exit 1

    tmpdir=/tmp/espnet-test-st
}


@test "translate_wav" {
    cd ./egs/fisher_callhome_spanish/st1/
    wav=../../../test_utils/st_test.wav
    model=fisher_callhome_spanish.transformer.v1.es-en
    ../../../utils/translate_wav.sh --python "coverage run --append" --stop-stage 2 --decode_dir $tmpdir --models ${model} ${wav}

    prefix="Translated text: "

    # NOTE(karita): If you will change the model, you should change these outputs.
    trans=$(../../../utils/translate_wav.sh --python "coverage run --append" --stage 3 --decode_dir ${tmpdir} --models ${model} ${wav} | grep "${prefix}")
    [ "$trans" = "${prefix}yes i'm jose" ]

    trans=$(../../../utils/translate_wav.sh --python "coverage run --append" --stage 3 --decode_dir ${tmpdir} --models ${model} --detokenize false ${wav} | grep "${prefix}")
    [ "$trans" = "${prefix}▁yes▁i▁&apos;m▁jose" ]
}
