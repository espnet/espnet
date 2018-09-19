# Note this isn't all the languages because I've already trained some: Missing 
# Bengali and Georgian
langs=(haitian kurmanji lao pashto swahili tagalog tamil
       tokpisin turkish vietnamese zulu)

for lang in ${langs[*]}; do
    echo ${lang}
    lang_recipe=${lang}_phoneme_objective
    #rsync -av --exclude="exp/" --exclude="dump/" \
    #        mono_phoneme_objective/ ${lang_recipe}
    #cd ${lang_recipe}
    ./run.sh --ngpu 1 --stage 1 --train-lang ${lang} \
            --mtlalpha 0.5 --phoneme-objective-weight 0.0
    ./run.sh --ngpu 1 --stage 3 --train-lang ${lang} \
            --mtlalpha 0.33 --phoneme-objective-weight 0.33 \
            --phoneme-objective-layer 2
done
