[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
    echo "Usage: local/score_sclite.sh <data-dir>"
    exit 1;
fi

dir=$1

ref=${dir}/ref.wrd.trn
hyp=${dir}/hyp.wrd.trn

# A:clean B:noisy C:clean with channel distortion D:noisy with channel distortion
cat ${ref} | grep '0)$' > ${dir}/ref.wrd_A.trn
cat ${hyp} | grep '0)$' > ${dir}/hyp.wrd_A.trn
cat ${ref} | grep '[123456])$' > ${dir}/ref.wrd_B.trn                        |
cat ${hyp} | grep '[123456])$' > ${dir}/hyp.wrd_B.trn
cat ${ref} | grep '7)$' > ${dir}/ref.wrd_C.trn                        |
cat ${hyp} | grep '7)$' > ${dir}/hyp.wrd_C.trn
cat ${ref} | grep '[89abcd])$' > ${dir}/ref.wrd_D.trn                        |
cat ${hyp} | grep '[89abcd])$' > ${dir}/hyp.wrd_D.trn

sclite -r ${dir}/ref.wrd_A.trn trn -h ${dir}/hyp.wrd_A.trn trn -i rm -o all stdout > ${dir}/result.wrd_A.txt
sclite -r ${dir}/ref.wrd_B.trn trn -h ${dir}/hyp.wrd_B.trn trn -i rm -o all stdout > ${dir}/result.wrd_B.txt
sclite -r ${dir}/ref.wrd_C.trn trn -h ${dir}/hyp.wrd_C.trn trn -i rm -o all stdout > ${dir}/result.wrd_C.txt
sclite -r ${dir}/ref.wrd_D.trn trn -h ${dir}/hyp.wrd_D.trn trn -i rm -o all stdout > ${dir}/result.wrd_D.txt

grep -e Avg -e SPKR -m 2 ${dir}/result.wrd_*.txt

exit 0
