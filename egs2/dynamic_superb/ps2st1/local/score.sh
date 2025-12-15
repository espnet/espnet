#!/bin/bash

ref=$1
hyp=$2
save_dir=$3

# Text normalization function
normalize_text() {
    local input_file=$1
    local output_file=$2

    # Apply text normalization: lowercase, remove punctuation, normalize whitespace
    awk '{
    # Extract utterance ID and text
    utt = $1
    $1 = ""
    text = substr($0, 2)

    # Convert to lowercase
    text = tolower(text)

    # Remove punctuation (keep only alphanumeric and spaces)
    gsub(/[^a-z0-9 ]/, "", text)

    # Normalize multiple spaces to single space
    gsub(/[ \t]+/, " ", text)

    # Remove leading/trailing whitespace
    gsub(/^[ \t]+|[ \t]+$/, "", text)

    # Print normalized text with utterance ID
    print utt " " text
    }' ${input_file} > ${output_file}
}


# Apply text normalization
echo "Normalizing reference and hypothesis texts..."
normalize_text ${ref} ${save_dir}/ref.normalized
normalize_text ${hyp} ${save_dir}/hyp.normalized

# Word Error Rate (using normalized texts)
# strip utterance IDs, keep only the text
awk '{ utt=$1; $1=""; print substr($0,2) " (spk-" utt ")" }' ${save_dir}/ref.normalized > ${save_dir}/ref.txt
awk '{ utt=$1; $1=""; print substr($0,2) " (spk-" utt ")" }' ${save_dir}/hyp.normalized > ${save_dir}/hyp.txt

# run sclite
sclite \
    -r ${save_dir}/ref.txt trn \
    -h ${save_dir}/hyp.txt trn \
    -i rm -o sum stdout \
    -F \
    | tee ${save_dir}/wer_sclite.log

echo "WER report saved to ${save_dir}/wer_sclite.log"

# Character Error Rate (using normalized texts)
# 1) Convert each line into "one char per token" plus the utt-ID:
awk '{ utt=$1; $1=""; txt=substr($0,2); gsub(/ /,"",txt);
    spaced=""; for(i=1;i<=length(txt);i++){spaced=spaced substr(txt,i,1)" ";}
    print spaced "(spk-" utt ")" }' ${save_dir}/ref.normalized \
    > ${save_dir}/ref.char.trn

awk '{ utt=$1; $1=""; txt=substr($0,2); gsub(/ /,"",txt);
    spaced=""; for(i=1;i<=length(txt);i++){spaced=spaced substr(txt,i,1)" ";}
    print spaced "(spk-" utt ")" }' ${save_dir}/hyp.normalized \
    > ${save_dir}/hyp.char.trn

# 2) Run sclite in transcription mode ("trn") but it's now character‐level:
sclite \
    -r ${save_dir}/ref.char.trn trn \
    -h ${save_dir}/hyp.char.trn trn \
    -i rm \
    -o sum stdout \
    | tee ${save_dir}/cer_sclite.log
echo "CER report saved to ${save_dir}/cer_sclite.log"
