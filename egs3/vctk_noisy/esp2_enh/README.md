# VCTK-Noisy esp2_enh recipe

## Data Preparation

Download VCTK-Noisy (VCTK-DEMAND) from
https://datashare.ed.ac.uk/handle/10283/2791 and set `VCTK_DEMAND` to the
dataset root. The root must contain these directories:

- `clean_trainset_28spk_wav/`
- `noisy_trainset_28spk_wav/`
- `clean_testset_wav/`
- `noisy_testset_wav/`

The recipe-local loader lives in `dataset/` (`VCTKNoisyDataset` and
`VCTKNoisyBuilder`). `create_dataset` only checks this layout; it does not
download the corpus. Speakers p226 and p287 form the validation split, the
other 26 training speakers form the training split, and `noisy_testset_wav`
is the test split. Audio is used at 16 kHz.

## Commands

```bash
cd egs3/vctk_noisy/esp2_enh
source path.sh

export VCTK_DEMAND=/path/to/vctk_noisy
python run.py --stages create_dataset \
    --training_config conf/training_conv_tasnet.yaml
python run.py --stages collect_stats \
    --training_config conf/training_conv_tasnet.yaml
python run.py --stages train \
    --training_config conf/training_conv_tasnet.yaml
python run.py --stages infer \
    --training_config conf/training_conv_tasnet.yaml \
    --inference_config conf/inference.yaml
python run.py --stages measure \
    --training_config conf/training_conv_tasnet.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
python run.py --stages pack_model \
    --training_config conf/training_conv_tasnet.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml \
    --publication_config conf/publication.yaml
```

To publish your own trained model, first set `model.model_file` in
`conf/inference.yaml` to the checkpoint you want to publish, then run `infer`,
`measure`, and `pack_model` with that config. Select a retained checkpoint:
the publication template excludes `last.ckpt` and `**/step*.ckpt`.

## Results

### Conv-TasNet

- Training config: `conf/training_conv_tasnet.yaml` (SI-SNR loss)
- Test set: 824 utterances

| Model | SI-SNR (dB) | PESQ | STOI |
| --- | ---: | ---: | ---: |
| Conv-TasNet | 19.61 | 2.58 | 0.940 |
| Noisy baseline | 8.45 | 1.968 | 0.921 |

Pretrained model: [espnet/vctk_noisy_esp2_enh_training_conv_tasnet](https://huggingface.co/espnet/vctk_noisy_esp2_enh_training_conv_tasnet)
