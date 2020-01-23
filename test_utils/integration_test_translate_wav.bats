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

    # TODO(karita): why tar results in error?
    ../../../utils/translate_wav.sh --stop-stage 2 --decode_dir $tmpdir --models fisher_callhome_spanish.transformer.v1.es-en ${wav}

    trans=$(../../../utils/translate_wav.sh --stage 3 --decode_dir $tmpdir --models fisher_callhome_spanish.transformer.v1.es-en ${wav} | grep "Translated text:")

    [ "$trans" = "Translated text: yes i'm jose" ]

    trans=$(../../../utils/translate_wav.sh --stage 3 --decode_dir $tmpdir --models fisher_callhome_spanish.transformer.v1.es-en --detokenize false ${wav} | grep "Translated text:")
    [ "$trans" = "Translated text: ▁yes▁i▁&apos;m▁jose" ]
}
