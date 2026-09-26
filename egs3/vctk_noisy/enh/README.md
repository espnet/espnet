# VCTK-Noisy Speech Enhancement Recipe (ESPnet3)

Speech enhancement recipe using Conv-TasNet trained on the VCTK-Noisy (VCTK-DEMAND) dataset.

## Dataset

Download VCTK-Noisy (VCTK-DEMAND) from:
https://datashare.ed.ac.uk/handle/10283/2791

Extract so that the following directories exist under your dataset root:
- `clean_trainset_28spk_wav/`
- `noisy_trainset_28spk_wav/`
- `clean_testset_wav/`
- `noisy_testset_wav/`

## Usage

```bash
cd egs3/vctk_noisy/enh
source path.sh

# Set dataset_dir in conf/training.yaml to your VCTK-Noisy root, then run:
python run.py --stages create_dataset --training_config conf/training.yaml
python run.py --stages collect_stats  --training_config conf/training.yaml
python run.py --stages train          --training_config conf/training.yaml
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

## Model

- Architecture: Conv-TasNet
- Dataset: VCTK-Noisy (VCTK-DEMAND)
- Sampling rate: 16 kHz
- Training split: 26 speakers (all except p226, p287)
- Validation split: p226, p287
- Test split: noisy_testset_wav

## Results

| Split | SI-SNR (dB) | PESQ  | STOI  |
|-------|-------------|-------|-------|
| test  | 18.73       | 2.551 | 0.937 |
| valid | 16.15       | 2.270 | 0.865 |
