
wav_dir=$1
data=$2

tmpdir=./tmp
mkdir -p $tmpdir

for x in spont_all; do
  [ ! -d $data/$x ] & mkdir -p $data/$x

  find $wav_dir/speech/spontaneous/ -iname "*.wav" > $tmpdir/all.wav
  find $wav_dir/trans/spontaneous/ -iname "*.txt" > $tmpdir/all.txt
  
  sed -e 's:.*/\(.*\)/\(.*\).wav$:\2 \1:' $tmpdir/all.wav | sort -k1,1 > $data/$x/utt2spk
  sed -e 's:.*/\(.*\)/.*/.*/\(.*\).wav$:\2 \1:' $tmpdir/all.wav | sort -k1,1 > $data/$x/utt2age
  sed -e 's:.*/\(.*\).wav$:\1:' $tmpdir/all.wav > $data/$x/all.uttids
  paste -d' ' $data/$x/all.uttids $tmpdir/all.wav | sort -k1,1 > $data/$x/wav.scp

  while read line; do
    [ -f $line ] || error_exit "Cannot find transcription file '$line'";
    head -n1 "$line" # | sed "s:<[^ ]*>::g" | sed "s:([^ ]*)::g" | sed "s:  : :g" | sed "s:  : :g" | sed "s:^ ::g" | sed "s:\[::g" | sed "s:\]::g" | sed "s:\*::g" | sed "s:^\xEF\xBB\xBF::g" | tr '[:lower:]' '[:upper:]'
  done < $tmpdir/all.txt > $tmpdir/trans
  paste -d' ' $data/$x/all.uttids $tmpdir/trans | sort -k1,1 > $data/$x/text
  dos2unix -n data/$x/text text_unix
  mv text_unix data/$x/text
  rm -f $data/$x/all.uttids

  spk2utt=$data/$x/spk2utt
  utils/utt2spk_to_spk2utt.pl < $data/$x/utt2spk >$spk2utt || exit 1

  age2utt=$data/$x/age2utt
  utils/utt2spk_to_spk2utt.pl < $data/$x/utt2age >$age2utt || exit 1

  ntrans=$(wc -l <$data/$x/text)
  nutt2spk=$(wc -l <$data/$x/utt2spk)
  ! [ "$ntrans" -eq "$nutt2spk" ] && \
    echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1;

  utils/validate_data_dir.sh --no-feats $data/$x || exit 1;
done
rm -r $tmpdir

echo "$0: successfully prepared data in $data"
exit 0
