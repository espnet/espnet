for x in test_st dev_st train_st_sp; do
  utils/fix_data_dir.sh data/${x}
  utils/data/get_utt2dur.sh data/${x}
  awk 'BEGIN{SUM=0}{SUM+=$2}END{print SUM/3600}' data/${x}/utt2dur
done
