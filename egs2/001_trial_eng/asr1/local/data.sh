cd /nlsasfs/home/nltm-st/akankss/espnet/egs2/001_hindi_all/asr1/
source /nlsasfs/home/nltm-st/akankss/miniconda3/bin/activate  # sources conda into the path
export PATH="/nlsasfs/home/nltm-st/akankss/local/bin/:$PATH" # to source local installations into the path
conda activate espnet # activating the conda env
mkdir -p data/train
mkdir -p data/dev
mkdir -p data/test
echo "#############GENERATING TRAIN FILES##########################"
python generate_files.py --wav-root "" --data-root "data/train" --df-path "/nlsasfs/home/nltm-st/akankss/datasets/dataset_hindi_complete/corrected/Hindi_combined_final"
echo "#############GENERATING DEV FILES##########################"
python generate_files.py --wav-root "" --data-root "data/dev" --df-path "/nlsasfs/home/nltm-st/akankss/datasets/dataset_hindi_complete/corrected/Hindi_combined_dev_final"
echo "#############GENERATING TEST FILES##########################"
python generate_files.py --wav-root "" --data-root "data/test" --df-path "/nlsasfs/home/nltm-st/akankss/datasets/dataset_hindi_complete/corrected/Hindi_combined_test_final"
echo "####################DONE WITH GENERATING FILES##############"
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
echo "*************************GENERATE data/train/spk2utt****************************"
head -n 10 data/train/spk2utt
echo "*************************GENERATE data/dev/spk2utt****************************"
head -n 10 data/dev/spk2utt
echo "*************************GENERATE data/test/spk2utt****************************"
head -n 10 data/test/spk2utt
echo "*************************TRAIN DIR****************************"
utils/validate_data_dir.sh --no-feats data/train
echo "*************************DEV DIR****************************"
utils/validate_data_dir.sh --no-feats data/dev
echo "*************************TEST DIR****************************"
utils/validate_data_dir.sh --no-feats data/test
echo "*************************TRAIN DIR****************************"
utils/fix_data_dir.sh data/train
echo "*************************DEV DIR****************************"
utils/fix_data_dir.sh data/dev
echo "*************************TEST DIR****************************"
utils/fix_data_dir.sh data/test