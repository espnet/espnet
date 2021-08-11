

echo "Data preparation"
local/download_data.sh raw_data
mkdir -p data
local/prepare_data.sh raw_data
local/check_audio_data_folder.sh raw_data
local/test_data_prep.sh raw_data data/test
local/train_data_prep.sh raw_data data/train
