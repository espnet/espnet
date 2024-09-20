# (1) To prepare the vocabulary
mkdir -p data/token_list/llm_vocab/
python3 ${python} pyscripts/utils/make_token_list_speechlm.py \
    --data_json "data/token_list/llm_vocab/data.json" \
    --token_list_dir "data/token_list/llm_vocab"

mkdir -p data/token_list/tts_vocab/
python3 ${python} pyscripts/utils/make_token_list_speechlm.py \
    --data_json "data/token_list/tts_vocab/data.json" \
    --token_list_dir "data/token_list/tts_vocab"

mkdir -p data/token_list/asr_vocab/
python3 ${python} pyscripts/utils/make_token_list_speechlm.py \
    --data_json "data/token_list/asr_vocab/data.json" \
    --token_list_dir "data/token_list/asr_vocab"
exit 0;

# (2) To prepare multiple data.json based on current TTS data.json
data_names="mls_en librispeech gigaspeech yodas_auto1 yodas_auto2 yodas_manual mls_multilingual emilia"
for data_name in $data_names; do
    src_dir=dump/raw_codec_ssl_tts_${data_name}

    # build ASR dataset
    tgt_dir=dump/raw_codec_ssl_asr_${data_name}
    for dset in `ls ${src_dir}`; do
        mkdir -p ${tgt_dir}/${dset}
        cp ${src_dir}/${dset}/{wav.scp,utt2spk,text} ${tgt_dir}/${dset}

        echo "building ${tgt_dir}/${dset}/data.json"
        python3 pyscripts/utils/make_speechlm_json.py \
            --task codec_ssl_asr \
            --output_json ${tgt_dir}/${dset}/data.json \
            --file_modality_type ${tgt_dir}/${dset}/wav.scp,codec_ssl,kaldi_ark \
            --file_modality_type ${tgt_dir}/${dset}/text,text_bpe,text
    done

    # build TTS dataset
    tgt_dir=dump/raw_tts_${data_name}
    for dset in `ls ${src_dir}`; do
        mkdir -p ${tgt_dir}/${dset}
        cp ${src_dir}/${dset}/{utt2spk,text} ${tgt_dir}/${dset}
        cp ${src_dir}/${dset}/codec_wav.scp ${tgt_dir}/${dset}/wav.scp

        echo "building ${tgt_dir}/${dset}/data.json"
        python3 pyscripts/utils/make_speechlm_json.py \
            --task tts \
            --output_json ${tgt_dir}/${dset}/data.json \
            --file_modality_type ${tgt_dir}/${dset}/wav.scp,codec,kaldi_ark \
            --file_modality_type ${tgt_dir}/${dset}/text,g2p,text \
            --file_modality_type ${tgt_dir}/${dset}/utt2spk,spk,text
    done

    # build AudioLM dataset
    tgt_dir=dump/raw_codec_ssl_audiolm_${data_name}
    for dset in `ls ${src_dir}`; do
        mkdir -p ${tgt_dir}/${dset}
        cp ${src_dir}/${dset}/wav.scp ${tgt_dir}/${dset}

        echo "building ${tgt_dir}/${dset}/data.json"
        python3 pyscripts/utils/make_speechlm_json.py \
            --task codec_ssl_audiolm \
            --output_json ${tgt_dir}/${dset}/data.json \
            --file_modality_type ${tgt_dir}/${dset}/wav.scp,codec_ssl,kaldi_ark
    done

    # build ssl_asr dataset
    tgt_dir=dump/raw_ssl_asr_${data_name}
    for dset in `ls ${src_dir}`; do
        mkdir -p ${tgt_dir}/${dset}
        cp ${src_dir}/${dset}/ssl_wav.scp ${tgt_dir}/${dset}/wav.scp
        cp ${src_dir}/${dset}/text ${tgt_dir}/${dset}

        echo "building ${tgt_dir}/${dset}/data.json"
        python3 pyscripts/utils/make_speechlm_json.py \
            --task ssl_asr \
            --output_json ${tgt_dir}/${dset}/data.json \
            --file_modality_type ${tgt_dir}/${dset}/wav.scp,ssl,kaldi_ark \
            --file_modality_type ${tgt_dir}/${dset}/text,text_bpe,text
    done
done
