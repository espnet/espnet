#!/bin/bash

cd /work/nvme/bbjs/sbharadwaj/icme_challenge/dump/fbank || exit 1

datasets=(7Msounds commonvoice ears mtg_jamendo vggsound yodas_speech)

export LC_ALL=C

# echo "merging data feats.scp"
# cat "${datasets[@]/%//feats.scp}" > train/feats.scp
# sort train/feats.scp -o train/feats.scp
# wc -l train/feats.scp
# echo "Checking for duplicates and sorted order"
# awk '{if (seen[$1]++) print "Duplicate utt_id:", $1}' train/feats.scp
# sort -c train/feats.scp

# echo "merging data spk2utt"
# cat "${datasets[@]/%//spk2utt}" > train/spk2utt
# sort train/spk2utt -o train/spk2utt
# wc -l train/spk2utt
# echo "Checking for duplicates and sorted order"
# awk '{if (seen[$1]++) print "Duplicate utt_id:", $1}' train/spk2utt
# sort -c train/spk2utt


# echo "merging data utt2spk"
# cat "${datasets[@]/%//utt2spk}" > train/utt2spk
# sort train/utt2spk -o train/utt2spk
# wc -l train/utt2spk
# echo "Checking for duplicates and sorted order"
# awk '{if (seen[$1]++) print "Duplicate utt_id:", $1}' train/utt2spk
# sort -c train/utt2spk


# echo "merging data utt2num_frames"
# cat "${datasets[@]/%//utt2num_frames}" > train/utt2num_frames
# sort train/utt2num_frames -o train/utt2num_frames
# wc -l train/utt2num_frames
# echo "Checking for duplicates and sorted order"
# awk '{if (seen[$1]++) print "Duplicate utt_id:", $1}' train/utt2num_frames
# sort -c train/utt2num_frames


# echo "merging data targets"
# find "${datasets[@]}" -maxdepth 1 -type f -name "target_iter3*" -exec cat {} + > train/target_iter3_beats_icme_inf
# sort train/target_iter3_beats_icme_inf -o train/target_iter3_beats_icme_inf
# wc -l train/target_iter3_beats_icme_inf
# echo "Checking for duplicates and sorted order"
# awk '{if (seen[$1]++) print "Duplicate utt_id:", $1}' train/target_iter3_beats_icme_inf
# sort -c train/target_iter3_beats_icme_inf

# id="1IWs-TIoSj4_001017731_001017892_spa_asr_segment0"
# for f in train/feats.scp train/spk2utt train/utt2spk train/utt2num_frames train/target_iter3_beats_icme_inf; do
#     grep -v "^${id}[[:space:]]" "$f" > "${f}.tmp" && mv "${f}.tmp" "$f"
# done
echo " Removing empty lines!"

awk 'NF==1{print $1}' train/target_iter3_beats_icme_inf > bad_utts.txt
for f in train/feats.scp train/utt2spk train/spk2utt train/utt2num_frames train/target_iter3_beats_icme_inf; do 
    grep -vFf bad_utts.txt "$f" > "${f}.filtered" && mv "${f}.filtered" "$f"; 
    wc -l $f
done