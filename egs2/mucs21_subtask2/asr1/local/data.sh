lang=hi-en

echo "Data preparation"
local/download_data.sh data $lang
if [ ! -e data/${lang}.path_done ]; then
  for dset in test train; do
      local/prepare_data.sh data/$lang/$dset/transcripts/wav.scp data/$lang/$dset/ out.scp
    done
  touch data/${lang}.path_done
  else
      echo "Path written already. Skipping."
  fi
