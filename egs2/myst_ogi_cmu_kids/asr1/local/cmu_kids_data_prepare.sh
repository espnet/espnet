# set -eu
corpus=$1
data=$2
lists_dir=$3  # Directory containing the train/dev/test lists

. ./path.sh
. ./utils/parse_options.sh

# Create the output directories
for d in $data/all; do
    mkdir -p $d
done

echo "Preparing cmu_kids..."
for kid in $corpus/kids/*; do
    if [ -d $kid ]; then
        spkID=$(basename $kid)
        sph="$kid/signal"
        if [ -d $sph ]; then
            for utt in $sph/*; do
                if [ ${utt: -4} == ".sph" ]; then
                    uttID=$(basename $utt)
                    uttID=${uttID%".sph"}
                    sentID=${uttID#$spkID}
                    sentID=${sentID:0:3}

                    # Find the sentence
                    sent=$(grep "$sentID" $corpus/tables/sentence.tbl | cut -f 3- | tr '[:lower:]' '[:upper:]' | tr -d '[:cntrl:]')

                    # Clean transcript
                    trans=$(tr -d '\n' < "$kid/trans/$uttID.trn" | tr '[:lower:]' '[:upper:]')

                    # Move all data to 'all'
                    echo "$uttID $spkID" >> $data/all/utt2spk
                    echo "$uttID sph2pipe -f wav -p -c 1 $utt|" >> $data/all/wav.scp
                    echo "$spkID "$(echo $spkID | cut -c 1)"" >> $data/all/spk2gender
                    echo "$uttID $sent" >> $data/all/text
                fi
            done
        fi
    fi

done
utils/fix_data_dir.sh $data/all

# Assuming the list files (train, dev, test) are in conf directory and have been pre-generated
for x in train dev test; do
  [ ! -d $data/$x ] && mkdir -p $data/$x

  # Filter the utt2spk, wav.scp, and text files based on the corresponding list
  grep -f $lists_dir/$x.list $data/all/utt2spk > $data/$x/utt2spk
  grep -f $lists_dir/$x.list $data/all/wav.scp > $data/$x/wav.scp
  grep -f $lists_dir/$x.list $data/all/text > $data/$x/text
#   grep -f $lists_dir/$x.list $data/all/spk2gender > $data/$x/spk2gender
  cut -d' ' -f2 $data/$x/utt2spk | sort | uniq | join -1 1 -2 1 - $data/all/spk2gender > $data/$x/spk2gender

  # Create the spk2utt mapping
  spk2utt=$data/$x/spk2utt
  utils/utt2spk_to_spk2utt.pl < $data/$x/utt2spk > $spk2utt || exit 1

  utils/fix_data_dir.sh $data/$x

  # Ensure the directory is valid
  utils/validate_data_dir.sh --no-feats $data/$x || exit 1
done

rm -rf $data/all
