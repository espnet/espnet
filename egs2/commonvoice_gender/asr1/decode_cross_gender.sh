#!/bin/bash
# Cross-gender decoding jobs

set -e
set -u
set -o pipefail

cd ~/bootcamp/espnet/egs2/commonvoice_gender/asr1

case "${1:-}" in
  male_on_female)
    # Use male-trained model, decode on female test set
    ./run.sh --stage 12 --stop-stage 13 --test_sets "test_female_en"
    ;;
  female_on_male)
    # Use female-trained model, decode on male test set
    ./run_female.sh --stage 12 --stop-stage 13 --test_sets "test_male_en"
    ;;
  *)
    echo "Usage: $0 {male_on_female|female_on_male}"
    exit 1
    ;;
esac

