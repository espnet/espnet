###########################################################################
# Create dictionaries with split diphthongs and standardized tones
###########################################################################
# In the lexicons provided by babel there are phonemes x_y, for which _y may
# or may not best be considered as a tag on phoneme x. In Lithuanian, for
# instance, there is a phoneme A_F for which _F or indicates failling tone.
# This same linguistic feature is represented in other languages as a "tag"
# (i.e. 判 pun3 p u: n _3), which means for the purposes of kaldi, that
# those phonemes share a root in the clustering decision tree, and the tag
# becomes an extra question. We may want to revisit this issue later.

dict=data/dict_universal

. ./utils/parse_options.sh
if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./local/prepare_dictionary.sh --dict data/dict_universal <lang_id>"
  exit 1
fi 

l=$1

mkdir -p $dict

echo "Making dictionary for ${l}"

# Create silence lexicon (This is the set of non-silence phones standardly
# used in the babel recipes
echo -e "<silence>\tSIL\n<unk>\t<oov>\n<noise>\t<sss>\n<v-noise>\t<vns>" \
  > ${dict}/silence_lexicon.txt

# Create non-silence lexicon
grep -vFf ${dict}/silence_lexicon.txt data/local/lexicon.txt \
  > data/local/nonsilence_lexicon.txt

# Create split diphthong and standarized tone lexicons for nonsilence words
./local/prepare_universal_lexicon.py \
  ${dict}/nonsilence_lexicon.txt data/local/nonsilence_lexicon.txt \
  local/phone_maps/${l} 

cat ${dict}/{,non}silence_lexicon.txt | sort > ${dict}/lexicon.txt

# Prepare the rest of the dictionary directory
# -----------------------------------------------
# The local/prepare_dict.py script, which is basically the same as
# prepare_unicode_lexicon.py used in the babel recipe to create the
# graphemic lexicons, is better suited for working with kaldi formatted
# lexicons and can be used for this task by only modifying optional input
# arguments. If we could modify local/prepare_lexicon.pl to accomodate this
# need it may be more intuitive.
./local/prepare_dict.py \
  --silence-lexicon ${dict}/silence_lexicon.txt ${dict}/lexicon.txt ${dict}

###########################################################################
# Prepend language ID to all utterances to disambiguate between speakers
# of different languages sharing the same speaker id.
#
# The individual lang directories can be used for alignments, while a
# combined directory will be used for training. This probably has minimal
# impact on performance as only words repeated across languages will pose
# problems and even amongst these, the main concern is the <hes> marker.
###########################################################################
echo "Prepend ${l} to data dir"
./utils/copy_data_dir.sh --spk-prefix "${l}_" --utt-prefix "${l}_" \
  data/train data/train_${l}
