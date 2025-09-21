#!/usr/bin/env bash
# Extract all datasets to organized data/ structure

set -e
set -u
set -o pipefail

# Create main data directory
mkdir -p data

echo "Extracting MUCS 2021 datasets..."
mkdir -p data/mucs2021
tar -xzf downloads/Hindi-English_train.tar.gz -C data/mucs2021/
tar -xzf downloads/Hindi-English_test.tar.gz -C data/mucs2021/

echo "Extracting LibriSpeech dataset..."
mkdir -p data/librispeech
tar -xzf downloads/train-clean-100.tar.gz -C data/librispeech/

echo "Extracting SLR103 Hindi datasets..."
mkdir -p data/slr103_hindi
tar -xzf downloads/Hindi_train.tar.gz -C data/slr103_hindi/
tar -xzf downloads/Hindi_test.tar.gz -C data/slr103_hindi/

echo "Dataset extraction complete!"
echo "Structure:"
find data/ -type d -maxdepth 3 | sort