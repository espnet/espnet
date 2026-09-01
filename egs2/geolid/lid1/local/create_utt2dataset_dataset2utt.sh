train_sets="
dump/raw/train_voxlingua107_lang
dump/raw/train_ml_superb2_lang
dump/raw/train_fleurs_lang
dump/raw/train_babel_over_10s_lang
dump/raw/train_voxpopuli_lang
"

train_all_dir="dump/raw/train_all_no_filter_lang_debug"

python local/create_utt2dataset.py \
    --train_sets "${train_sets}" \
    --train_all_dir "${train_all_dir}"

utils/utt2spk_to_spk2utt.pl "${train_all_dir}/utt2dataset" > "${train_all_dir}/dataset2utt"
