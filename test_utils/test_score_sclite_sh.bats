#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    cat << EOF > $tmpdir/data.1.json
{
	"utts": {
        "uttid": {
            "output": [
                [
                    {
                        "name": "target1[1]",
                        "rec_text": "ONE TWO THREE FOUR FIVE SIX<eos>",
                        "rec_token": "O N E <space> T W O <space> T H R E E <space> F O U R <space> F I V E <space> S I X <eos>",
                        "rec_tokenid": "15 14 5 27 20 23 15 27 20 8 18 5 5 27 6 15 21 18 27 6 9 22 5 27 19 9 24",
                        "score": -1.1091572046279907,
                        "shape": [
                            118,
                            52
                        ],
                        "text": "ONE TWO THREE FOUR FIVE SIX",
                        "token": "O N E <space> T W O <space> T H R E E <space> F O U R <space> F I V E <space> S I X",
                        "tokenid": "15 14 5 27 20 23 15 27 20 8 18 5 5 27 6 15 21 18 27 6 9 22 5 27 19 9 24"
                    }
                ],
				[
                    {
                        "name": "target2[1]",
                        "rec_text": "SIX FIVE FOUR THREE TWO ONE<eos>",
                        "rec_token": "S I X <space> F I V E <space> F O U R <space> T H R E E <space> T W O <space> O N E <eos>",
                        "rec_tokenid": "19 9 24 27 6 9 22 5 27 6 15 21 18 27 20 8 18 5 5 27 20 23 15 27 15 14 5",
                        "score": -0.9484919905662537,
                        "shape": [
                            128,
                            52
                        ],
                        "text": "SIX FIVE FOUR THREE TWO",
                        "token": "S I X <space> F I V E <space> F O U R <space> T H R E E <space> T W O",
                        "tokenid": "19 9 24 27 6 9 22 5 27 6 15 21 18 27 20 8 18 5 5 27 20 23 15"
                    }
                ]
            ],
            "utt2spk": "foobar"
        }
    }
}
EOF

    cat << EOF > $tmpdir/dictionary.txt
A 1
B 2
C 3
D 4
E 5
F 6
G 7
H 8
I 9
J 10
K 11
L 12
M 13
N 14
O 15
P 16
Q 17
R 18
S 19
T 20
U 21
V 22
W 23
X 24
Y 26
Z 26
<space> 27
EOF

    cat << EOF > $tmpdir/valid_min_perm_result_json
Total Scores: (#C #S #D #I) 50 0 0 4
Error Rate:   8.00
Total Utts:  1
{
    "utts": {
        "(foobar-uttid)": {
            "Scores": "(#C #S #D #I) 50 0 0 4",
            "r1h1": {
                "HYP": "o n e <space> t w o <space> t h r e e <space> f o u r <space> f i v e <space> s i x",
                "REF": "o n e <space> t w o <space> t h r e e <space> f o u r <space> f i v e <space> s i x",
                "Scores": "(#C #S #D #I) 27 0 0 0",
                "Speaker": "sentences 0: foobar #utts: 1"
            },
            "r2h2": {
                "HYP": "s i x <space> f i v e <space> f o u r <space> t h r e e <space> t w O <SPACE> o N E",
                "REF": "s i x <space> f i v e <space> f o u r <space> t h r e e <space> t w * ******* o * *",
                "Scores": "(#C #S #D #I) 23 0 0 4",
                "Speaker": "sentences 0: foobar #utts: 1"
            }
        }
    }
}
EOF

    cat << EOF > $tmpdir/valid_min_perm_result_wrd_json
Total Scores: (#C #S #D #I) 11 0 0 1
Error Rate:   9.09
Total Utts:  1
{
    "utts": {
        "(foobar-uttid)": {
            "Scores": "(#C #S #D #I) 11 0 0 1",
            "r1h1": {
                "HYP": "one two three four five six",
                "REF": "one two three four five six",
                "Scores": "(#C #S #D #I) 6 0 0 0",
                "Speaker": "sentences 0: foobar #utts: 1"
            },
            "r2h2": {
                "HYP": "six five four three two ONE",
                "REF": "six five four three two ***",
                "Scores": "(#C #S #D #I) 5 0 0 1",
                "Speaker": "sentences 0: foobar #utts: 1"
            }
        }
    }
}
EOF
}

teardown() {
    rm -rf $tmpdir
}

@test "" {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils

    sed "s=MAIN_ROOT\=\$PWD/../../..=MAIN_ROOT\=$(cd $BATS_TEST_DIRNAME/..; pwd)=" \
        < $(cd $BATS_TEST_DIRNAME/..; pwd)/egs/wsj/asr1/path.sh > path.sh
    ln -s $(cd $BATS_TEST_DIRNAME/..; pwd)/egs/wsj/asr1/utils/parse_options.sh ./utils

    ${utils}/score_sclite.sh --wer true --num_spkrs 2 ${tmpdir} ${tmpdir}/dictionary.txt
    sed -i '1d' ${tmpdir}/min_perm_result.json
    diff ${tmpdir}/min_perm_result.json ${tmpdir}/valid_min_perm_result_json
    sed -i '1d' ${tmpdir}/min_perm_result.wrd.json
    diff ${tmpdir}/min_perm_result.wrd.json ${tmpdir}/valid_min_perm_result_wrd_json
    rm path.sh
    rm utils/parse_options.sh
}
