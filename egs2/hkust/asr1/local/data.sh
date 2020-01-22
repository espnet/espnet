#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
}
help_message=$(cat << EOF
Usage: $0
EOF
)
SECONDS=0

log "$0 $*"

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
  log "${help_message}"
  log "Error: No positional arguments are required."
  exit 2
fi

if [ -z "${HKUST1}" ]; then
    log "Error: \$HKUST1 is not set in db.sh."
    exit 2
fi
if [ -z "${HKUST2}" ]; then
    log "Error: \$HKUST2 is not set in db.sh."
    exit 2
fi

hkust_audio_dir=$HKUST1
hkust_text_dir=$HKUST2

train_dir=data/local/train
dev_dir=data/local/dev

mkdir -p $train_dir
mkdir -p $dev_dir

#data directory check
if [ ! -d $hkust_audio_dir ] || [ ! -d $hkust_text_dir ]; then
  log "Error: $0 requires two directory arguments"
  exit 1;
fi

log "Data Preparation"
#find sph audio file for train dev resp.
find $hkust_audio_dir -iname "*.sph" | grep -i "audio/train" > $train_dir/sph.flist || exit 1;
find $hkust_audio_dir -iname "*.sph" | grep -i "audio/dev" > $dev_dir/sph.flist || exit 1;

n=`cat $train_dir/sph.flist $dev_dir/sph.flist | wc -l`
[ $n -ne 897 ] && \
  log Warning: expected 897 data data files, found $n

#Transcriptions preparation

#collect all trans, convert encodings to utf-8,
find $hkust_text_dir -iname "*.txt" | grep -i "trans/train" | xargs cat |\
  iconv -f GBK -t utf-8 - | perl -e '
    while (<STDIN>) {
      @A = split(" ", $_);
      if (@A <= 1) { next; }
      if ($A[0] eq "#") { $utt_id = $A[1]; }
      if (@A >= 3) {
        $A[2] =~ s:^([AB])\:$:$1:;
        printf "%s-%s-%06.0f-%06.0f", $utt_id, $A[2], 100*$A[0] + 0.5, 100*$A[1] + 0.5;
        for($n = 3; $n < @A; $n++) { print " $A[$n]" };
        print "\n";
      }
    }
  ' | sort -k1 > $train_dir/transcripts.txt || exit 1;

find $hkust_text_dir -iname "*.txt" | grep -i "trans/dev" | xargs cat |\
  iconv -f GBK -t utf-8 - | perl -e '
    while (<STDIN>) {
      @A = split(" ", $_);
      if (@A <= 1) { next; }
      if ($A[0] eq "#") { $utt_id = $A[1]; }
      if (@A >= 3) {
        $A[2] =~ s:^([AB])\:$:$1:;
        printf "%s-%s-%06.0f-%06.0f", $utt_id, $A[2], 100*$A[0] + 0.5, 100*$A[1] + 0.5;
        for($n = 3; $n < @A; $n++) { print " $A[$n]" };
        print "\n";
      }
    }
  ' | sort -k1  > $dev_dir/transcripts.txt || exit 1;

#transcripts normalization and segmentation
#(this needs external tools),
python -c "import mmseg" 2>/dev/null || \
  (log "mmseg is not found. Checkout tools/extra/install_mmseg.sh" && exit 1;)

cat $train_dir/transcripts.txt |\
  sed -e 's/<foreign language=\"[a-zA-Z]\+\">/ /g' |\
  sed -e 's/<\/foreign>/ /g' |\
  sed -e 's/<noise>\(.\+\)<\/noise>/\1/g' |\
  sed -e 's/((\([^)]\{0,\}\)))/\1/g' |\
  local/hkust_normalize.pl |\
  python local/hkust_segment.py |\
  awk '{if (NF > 1) print $0;}' > $train_dir/text || exit 1;

cat $dev_dir/transcripts.txt |\
  sed -e 's/<foreign language=\"[a-zA-Z]\+\">/ /g' |\
  sed -e 's/<\/foreign>/ /g' |\
  sed -e 's/<noise>\(.\+\)<\/noise>/\1/g' |\
  sed -e 's/((\([^)]\{0,\}\)))/\1/g' |\
  local/hkust_normalize.pl |\
  python local/hkust_segment.py |\
  awk '{if (NF > 1) print $0;}' > $dev_dir/text || exit 1;

# some data is corrupted. Delete them
cat $train_dir/text | grep -v 20040527_210939_A901153_B901154-A-035691-035691 | egrep -v "A:|B:" > tmp
mv tmp $train_dir/text || exit 1;

#Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#sw02001-A_000098-001156 sw02001-A 0.98 11.56


awk '{ segment=$1; split(segment,S,"-"); side=S[2]; audioname=S[1];startf=S[3];endf=S[4];
   print segment " " audioname "-" side " " startf/100 " " endf/100}' <$train_dir/text > $train_dir/segments
awk '{name = $0; gsub(".sph$","",name); gsub(".*/","",name); print(name " " $0)}' $train_dir/sph.flist > $train_dir/sph.scp

awk '{ segment=$1; split(segment,S,"-"); side=S[2]; audioname=S[1];startf=S[3];endf=S[4];
   print segment " " audioname "-" side " " startf/100 " " endf/100}' <$dev_dir/text > $dev_dir/segments
awk '{name = $0; gsub(".sph$","",name); gsub(".*/","",name); print(name " " $0)}' $dev_dir/sph.flist > $dev_dir/sph.scp

sph2pipe=`which sph2pipe` || sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
[ ! -x $sph2pipe ] && log "Could not find the sph2pipe program at $sph2pipe" && exit 1;

cat $train_dir/sph.scp | awk -v sph2pipe=$sph2pipe '{printf("%s-A %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2);
    printf("%s-B %s -f wav -p -c 2 %s |\n", $1, sph2pipe, $2);}' | \
   sort > $train_dir/wav.scp || exit 1;

cat $dev_dir/sph.scp | awk -v sph2pipe=$sph2pipe '{printf("%s-A %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2);
    printf("%s-B %s -f wav -p -c 2 %s |\n", $1, sph2pipe, $2);}' | \
   sort > $dev_dir/wav.scp || exit 1;
#side A - channel 1, side B - channel 2

# this file reco2file_and_channel maps recording-id (e.g. sw02001-A)
# to the file name sw02001 and the A, e.g.
# sw02001-A  sw02001 A
# In this case it's trivial, but in other corpora the information might
# be less obvious.  Later it will be needed for ctm scoring.
cat $train_dir/wav.scp | awk '{print $1}' | \
  perl -ane '$_ =~ m:^(\S+)-([AB])$: || die "bad label $_"; print "$1-$2 $1 $2\n"; ' \
  > $train_dir/reco2file_and_channel || exit 1;
cat $dev_dir/wav.scp | awk '{print $1}' | \
  perl -ane '$_ =~ m:^(\S+)-([AB])$: || die "bad label $_"; print "$1-$2 $1 $2\n"; ' \
  > $dev_dir/reco2file_and_channel || exit 1;


cat $train_dir/segments | awk '{spk=substr($1,1,33); print $1 " " spk}' > $train_dir/utt2spk || exit 1;
cat $train_dir/utt2spk | sort -k 2 | utils/utt2spk_to_spk2utt.pl > $train_dir/spk2utt || exit 1;

cat $dev_dir/segments | awk '{spk=substr($1,1,33); print $1 " " spk}' > $dev_dir/utt2spk || exit 1;
cat $dev_dir/utt2spk | sort -k 2 | utils/utt2spk_to_spk2utt.pl > $dev_dir/spk2utt || exit 1;


log "Formatting Data Directory"

mkdir -p data/train data/dev

# Copy stuff into its final locations...

for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp data/local/train/$f data/train/$f || exit 1;
done

for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp data/local/dev/$f data/dev/$f || exit 1;
done

log "hkust_format_data succeeded."


log "Upsample audios from 8k to 16k"
# upsample audio from 8k to 16k to make a recipe consistent with others
for x in train dev; do
  sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
done
# remove space in text
for x in train dev; do
  cp data/${x}/text data/${x}/text.org
  paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
    > data/${x}/text
  rm data/${x}/text.org
done

log "Successfully finished. [elapsed=${SECONDS}s]"
exit 0
