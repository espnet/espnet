lang=hi-en

echo "Data preparation"
local/download_data.sh data $lang

for dset in test train; do
    local/prepare_data.sh data/$lang/$dset/transcripts/wav.scp data/$lang/$dset/ out.scp
done
