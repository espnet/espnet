#!/usr/bin/env bats
# -*- mode:sh -*-

setup() {
    export LC_ALL="en_US.UTF-8"
    tmpdir=$(mktemp -d test_spm.XXXXXX)
    echo $tmpdir
}

teardown() {
    rm -rf $tmpdir
}

@test "spm_xxx" {
    testfile=test/tedlium2.txt
    nbpe=100
    bpemode=unigram
    bpemodel=$tmpdir/test_spm

    spm_train --input=${testfile} --vocab_size=${nbpe} --model_type=${bpemode} \
          --model_prefix=${bpemodel} --input_sentence_size=100000000 \
          --character_coverage=1.0 --bos_id=-1 --eos_id=-1 \
          --unk_id=0 --user_defined_symbols=[laughter],[noise],[vocalized-noise]

    diff ${bpemodel}.vocab test/tedlium2.vocab

    txt="test sentencepiece.[noise]"

    enc=$(echo $txt | spm_encode --model=${bpemodel}.model --output_format=piece)
    [ "$enc" = "▁ te s t ▁ s en t en c e p ie c e . [noise]" ]

    dec=$(echo $enc | spm_decode --model=${bpemodel}.model --input_format=piece)
    [ "$dec" = "$txt" ]
}
