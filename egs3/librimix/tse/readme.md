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
