# LibriMix TSE recipe

Place the corpus under `data/LibriSpeech`, or set the `LIBRISPEECH`
environment variable to an existing LibriSpeech root, before running
`create_dataset`/`train`.

## Quick start

```bash
# 0) Load Python environment
source path.sh

#-----------------------------------------
# Option 1: Run all/multiple stages in one command
#-----------------------------------------
python run.py \
    --stages all \
    --training_config conf/tuning/training_td_speakerbeam_16k.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml \
    --publication_config conf/publication.yaml \
    --demo_config conf/demo.yaml
# You can also run specific stages, e.g., `--stages train infer measure`.

#-----------------------------------------
# Option 2: Run each stage separately
#-----------------------------------------
# 1) Prepare the dataset
# If the dataset has already been created, please set the `LIBRIMIX`
#    environment variable to the existing LibriMix root.
# Otherwise, the dataset will be created under `data/LibriMix`.
python run.py \
    --stages create_dataset

# 2) Collect statistics for normalization
python run.py \
    --stages collect_stats \
    --training_config conf/tuning/training_td_speakerbeam_16k.yaml

# 3) Train the default TD-SpeakerBeam model
python run.py \
    --stages train \
    --training_config conf/tuning/training_td_speakerbeam_16k.yaml

# 4) Infer to generate extracted speech
python run.py --stages infer \
    --training_config conf/tuning/training_td_speakerbeam_16k.yaml \
    --inference_config conf/inference.yaml

# 5) Score to evaluate the extracted speech
python run.py --stages measure \
    --training_config conf/tuning/training_td_speakerbeam_16k.yaml \
    --metrics_config conf/metrics.yaml

# 6) Pack the model for publication
python run.py --stages pack_model \
    --training_config conf/tuning/training_td_speakerbeam_16k.yaml \
    --publication_config conf/publication.yaml

# 7) Create a demo for the model
# ------------------------------
# Note: You might want to manually remove 'speech_ref1' from the 'input_key' of
#       `conf/inference.yaml` in the `model_pack` directory for deploying the demo
#       without requiring reference speech.
# ------------------------------
python run.py --stages pack_demo \
    --training_config conf/tuning/training_td_speakerbeam_16k.yaml \
    --demo_config conf/demo.yaml

# 8) Upload the model/demo to Hugging Face Hub
python run.py --stages upload_model upload_demo \
    --training_config conf/tuning/training_td_speakerbeam_16k.yaml \
    --publication_config conf/publication.yaml \
    --demo_config conf/demo.yaml
```

## Pretrained Models

### TD-SpeakerBeam 16kHz (trained on Libri2Mix mix-clean data)

 - config: conf/tuning/training_td_speakerbeam_16k.yaml
 - Pretrained model: https://huggingface.co/wyz/librimix_tse_clean_2mix_training_td_speakerbeam_16k

|dataset|[PESQ_WB](https://github.com/espnet/espnet/blob/master/espnet3/systems/tse/metrics/pesq.py) ↑|[ESTOI](https://github.com/espnet/espnet/blob/master/espnet3/systems/tse/metrics/stoi.py) (×100)↑|[SDR](https://github.com/espnet/espnet/blob/master/espnet3/systems/tse/metrics/sdr.py) (dB)↑|[SAR](https://github.com/espnet/espnet/blob/master/espnet3/systems/tse/metrics/sdr.py) (dB)↑|[SIR](https://github.com/espnet/espnet/blob/master/espnet3/systems/tse/metrics/sdr.py) (dB)↑|[SI-SNR](https://github.com/espnet/espnet/blob/master/espnet3/systems/tse/metrics/sisnr.py) (dB)↑|[OVRL](https://github.com/espnet/espnet/blob/master/espnet3/systems/tse/metrics/dnsmos.py)↑|[SIG](https://github.com/espnet/espnet/blob/master/espnet3/systems/tse/metrics/dnsmos.py)↑|[BAK](https://github.com/espnet/espnet/blob/master/espnet3/systems/tse/metrics/dnsmos.py)↑|[P808_MOS](https://github.com/espnet/espnet/blob/master/espnet3/systems/tse/metrics/dnsmos.py)↑|[TSOS](https://github.com/espnet/espnet/blob/master/espnet3/systems/tse/metrics/tsos.py) (%)↓|
|---|---|---|---|---|---|---|---|---|---|---|---|
|2mix_16k_max_dev_mix-clean|2.21|82.86|13.29|13.30|49.87|12.66|3.04|3.38|3.92|3.37|7.49|
|2mix_16k_max_test_mix-clean|2.20|83.06|13.27|13.27|49.99|12.66|3.06|3.40|3.94|3.39|6.26|
