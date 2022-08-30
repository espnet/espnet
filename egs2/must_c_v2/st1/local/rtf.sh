src=$1
nj=200

dst=$src/rtf.log

rm $dst

for n in $( seq 1 $nj ); do
    grep "RTF" $src/logdir/st_inference.${n}.log >> $dst
done

python local/rtf.py --src $dst