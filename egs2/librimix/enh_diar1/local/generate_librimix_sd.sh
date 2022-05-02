#!/bin/env bash
set -eu  # Exit on error

storage_dir=$1
folder=$2
fs=$3
min_max_mode=$4
librispeech_dir=$storage_dir/LibriSpeech
wham_dir=$storage_dir/wham_noise
librimix_outdir=$storage_dir/

function LibriSpeech_dev_clean() {
	if ! test -e $librispeech_dir/dev-clean; then
		echo "Download LibriSpeech/dev-clean into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/dev-clean.tar.gz -P $storage_dir
		tar -xzf $storage_dir/dev-clean.tar.gz -C $storage_dir
		rm -rf $storage_dir/dev-clean.tar.gz
	fi
}

function LibriSpeech_test_clean() {
	if ! test -e $librispeech_dir/test-clean; then
		echo "Download LibriSpeech/test-clean into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/test-clean.tar.gz -P $storage_dir
		tar -xzf $storage_dir/test-clean.tar.gz -C $storage_dir
		rm -rf $storage_dir/test-clean.tar.gz
	fi
}

function LibriSpeech_clean100() {
	if ! test -e $librispeech_dir/train-clean-100; then
		echo "Download LibriSpeech/train-clean-100 into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-100.tar.gz -P $storage_dir
		tar -xzf $storage_dir/train-clean-100.tar.gz -C $storage_dir
		rm -rf $storage_dir/train-clean-100.tar.gz
	fi
}

function LibriSpeech_clean360() {
	if ! test -e $librispeech_dir/train-clean-360; then
		echo "Download LibriSpeech/train-clean-360 into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-360.tar.gz -P $storage_dir
		tar -xzf $storage_dir/train-clean-360.tar.gz -C $storage_dir
		rm -rf $storage_dir/train-clean-360.tar.gz
	fi
}

function wham() {
	if ! test -e $wham_dir; then
		echo "Download wham_noise into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 https://storage.googleapis.com/whisper-public/wham_noise.zip -P $storage_dir
		unzip -qn $storage_dir/wham_noise.zip -d $storage_dir
		rm -rf $storage_dir/wham_noise.zip
	fi
}

LibriSpeech_dev_clean &
LibriSpeech_test_clean &
LibriSpeech_clean100 &
wham &

wait

# Path to python
python_path=python3

# If you wish to rerun this script in the future please comment this line out.
#$python_path $folder/scripts/augment_train_noise.py --wham_dir $wham_dir

for n_src in 2 3; do
  metadata_dir=$folder/metadata/Libri$n_src"Mix"
  $python_path $folder/scripts/create_librimix_from_metadata.py --librispeech_dir $librispeech_dir \
    --wham_dir $wham_dir \
    --metadata_dir $metadata_dir \
    --librimix_outdir $librimix_outdir \
    --n_src $n_src \
    --freqs $fs \
    --modes $min_max_mode \
    --types mix_both
done