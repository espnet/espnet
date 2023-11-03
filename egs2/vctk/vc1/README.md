# VCTK RECIPE

This is the recipe of the English multi-speaker TTS model with [VCTK](http://www.udialogue.org/download/cstr-vctk-corpus.html) corpus.


```sh
# Prep data directory
./run.sh --stage 1 --stop-stage 1
```
Data prepare from stage 2 to stage 5 (for non-parallel training data, using x-vector, without text transcripts).
```sh
# Run from stage 2 to stage 5
./run.sh \
    --train_set tr_no_dev_phn \
    --valid_set dev_phn \
    --test_sets "dev_phn eval1_phn" \
    --srctexts "data/tr_no_dev_phn/text" \
    --g2p none \
    --cleaner none \
    --token_type none \
    --use_xvector true \
    --stage 2 \
    --stop_stage 5 \
    --use_sid false \
    --min_wav_duration 0.38 \
    --ngpu 1 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --dumpdir dump/22k \
    --expdir exp/22k \
    --win_length null \
    --vc_task non_parallel_vc \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_vc_spk_autovc.yaml \
    --inference_model train.total_count.ave.pth
```
