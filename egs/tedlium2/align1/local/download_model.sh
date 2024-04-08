#!/usr/bin/env bash


dir=${download_dir}/${models}
mkdir -p ${dir}

function download_models () {
    if [ -z $models ]; then
        return
    fi

    file_ext="tar.gz"
    case "${models}" in
        "tedlium2.rnn.v1") share_url="https://drive.google.com/open?id=1UqIY6WJMZ4sxNxSugUqp3mrGb3j6h7xe"; api=v1 ;;
        "tedlium2.rnn.v2") share_url="https://drive.google.com/open?id=1cac5Uc09lJrCYfWkLQsF8eapQcxZnYdf"; api=v1 ;;
        "tedlium2.transformer.v1") share_url="https://drive.google.com/open?id=1cVeSOYY1twOfL9Gns7Z3ZDnkrJqNwPow" ;;
        "tedlium3.transformer.v1") share_url="https://drive.google.com/open?id=1zcPglHAKILwVgfACoMWWERiyIquzSYuU" ;;
        "librispeech.transformer.v1") share_url="https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6" ;;
        "librispeech.transformer.v1.transformerlm.v1") share_url="https://drive.google.com/open?id=1RHYAhcnlKz08amATrf0ZOWFLzoQphtoc" ;;
        "commonvoice.transformer.v1") share_url="https://drive.google.com/open?id=1tWccl6aYU67kbtkm8jv5H6xayqg1rzjh" ;;
        "csj.transformer.v1") share_url="https://drive.google.com/open?id=120nUQcSsKeY5dpyMWw_kI33ooMRGT2uF" ;;
        "wsj.transformer.v1") share_url="https://drive.google.com/open?id=1Az-4H25uwnEFa4lENc-EKiPaWXaijcJp" ;;
        "wsj.transformer_small.v1") share_url="https://drive.google.com/open?id=1jdEKbgWhLTxN_qP4xwE7mTOPmp7Ga--T" ;;
        *) echo "No such models: ${models}"; exit 1 ;;
    esac

    if [ ! -e ${dir}/.complete ]; then
        download_from_google_drive.sh ${share_url} ${dir} ${file_ext}
        touch ${dir}/.complete
    fi
}

download_models
