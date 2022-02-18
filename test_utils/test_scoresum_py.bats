#!/usr/bin/env bats

setup() {
    . tools/activate_python.sh 
    tools/installers/install_longformer.sh 
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/egs2/TEMPLATE/asr1/pyscripts/utils/
    export LC_ALL="en_US.UTF-8"
    tmpdir=$(mktemp -d testXXXXXX)/
    mkdir -p $tmpdir/valid/ $tmpdir/test
    echo $tmpdir
    cat <<EOF > $tmpdir/valid/hyp
BAGGY EYES CAN HELP INCREASE THE CIRCULATION AND REDUCE THE SWELLING . FIND OUT HOW TO TREAT BAGGY EYES WITH TIPS FROM A PROFESSIONAL MAKEUP ARTIST IN THIS FREE VIDEO ON SKIN CARE .
KICKBOXERS USE A POINTED TOE TO DEFEND AGAINST A LOW KICK . LEARN HOW TO DO A LOW KICK IN THIS FREE VIDEO ON WOMEN'S KICKBOXING TECHNIQUES .
EOF
    cp $tmpdir/valid/hyp $tmpdir/valid/ref
    cat <<EOF > $tmpdir/test/hyp
eg1 BAGGY EYES CAN HELP INCREASE THE CIRCULATION AND REDUCE THE SWELLING . FIND OUT HOW TO TREAT BAGGY EYES WITH TIPS FROM A PROFESSIONAL MAKEUP ARTIST IN THIS FREE VIDEO ON SKIN CARE .
eg2 KICKBOXERS USE A POINTED TOE TO DEFEND AGAINST A LOW KICK . LEARN HOW TO DO A LOW KICK IN THIS FREE VIDEO ON WOMEN 'S KICKBOXING TECHNIQUES .
EOF
    cat << EOF > $tmpdir/test/ref
eg1 TO TREAT BAGGY EYES, MASSAGE SKIN-CARE PRODUCTS ON EYES AND DRINK PLENTY OF WATER. REMOVE BAGGY EYES WITH TIPS FROM A PROFESSIONAL MAKEUP ARTIST IN THIS FREE VIDEO ABOUT THE BASICS OF SKIN CARE .
eg2 MIXED MARTIAL ARTISTS USE THE FRONT KICK TO STRIKE AN OPPONENT'S STOMACH OR FACE. LEARN HOW TO DO A BACK LEG FRONT KICK IN THIS FREE VIDEO ON WOMEN'S KICKBOXING TECHNIQUES.
EOF

cat << EOF > $tmpdir/result_dev.txt
Key      METEOR          ROUGE-L
eg1      1.0     1.0
eg2      1.0     1.0
RESULT 100.0 100.0 100.0 100.0 100.0
EOF

cat << EOF > $tmpdir/result_test.txt
Key      METEOR          ROUGE-L
eg1      0.30719663705747413     0.5492890995260664
eg2      0.27705756268835646     0.4638783269961977
RESULT 62.83582089552239 47.02917771883289 55.199004975124375 29.24673171958248 90.85839986801147
EOF
}

teardown() {
    rm -rf $tmpdir
}

@test "score_summarization.py" {
    python ${utils}/score_summarization.py ${tmpdir}/ref $tmpdir/valid/hyp > ${tmpdir}/output.txt
    diff ${tmpdir}/result_dev.txt ${tmpdir}/output.txt
    python ${utils}/score_summarization.py ${tmpdir}/test/ref ${tmpdir}test/hyp > ${tmpdir}/output.txt
    diff ${tmpdir}/result_test.txt ${tmpdir}/output.txt
}
