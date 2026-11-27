# LibriTTS LACE recipe

LACE (Layer-Adaptive Codec Encoding) fine-tuning of a neural codec on LibriTTS, the ESPnet3 recipe of [LACE: Layer-Wise Compression for Dynamic Frame Rate Codecs](https://arxiv.org/abs/2609.17509) (SLT 2026).
LACE wraps the residual vector quantizer of a trained codec so that, before every codebook layer, similar neighbouring frames are merged into segments and each layer runs at its own, lower frame rate.
The compression `rate` controls how much is merged: a higher rate keeps more segments and compresses less.
The model code lives in `espnet3/systems/lace`; the default configuration fine-tunes the LibriTTS EnCodec published from the ESPnet3 codec recipe, [`espnet/libritts_esp2_codec_train_encodec_libritts`](https://huggingface.co/espnet/libritts_esp2_codec_train_encodec_libritts).

## Data Preparation

On the first run `create_dataset` downloads the five LibriTTS subsets into `downloads/LibriTTS`, roughly 80 GB, and writes the `train`/`valid`/`test` manifests under `data/manifest`:

```bash
python run.py --stages create_dataset --training_config conf/training.yaml
```

If LibriTTS is already on disk, point the recipe at it rather than downloading a second copy: set `builder.dataset_path` in `dataset/config.yaml` to the directory that holds the `LibriTTS` folder.
An absolute path is used exactly as written, and a relative one is resolved from the recipe directory.
A subset counts as prepared only once it carries a `.complete` marker, which `create_dataset` writes after extracting that subset; a corpus staged any other way needs the markers created by hand (`touch <dataset_path>/LibriTTS/<subset>/.complete`), or the recipe downloads the subsets again.

The codec that is fine-tuned is a published ESPnet3 model bundle (the `pack_model` output of the codec recipe, uploaded by its `upload_model` stage).
`model.pretrained_model_tag` names it by its Hugging Face id, and `espnet_model_zoo` downloads and caches it on first use.
An espnet2 zoo model (`meta.yaml` with `train_config`) is accepted the same way, as long as its checkpoint matches the current espnet2 code.
To fine-tune a codec you trained with an ESPnet3 codec recipe instead, run its `pack_model` stage and set `model.pretrained_model_dir` to the resulting `model_pack` directory (or `pretrained_model_tag` to the repo its `upload_model` stage pushed to).

## Commands

Run the following commands from `egs3/libritts/lace/`.
`path.sh` activates the tools environment and puts this checkout on `PYTHONPATH`.

```bash
. ./path.sh

# 1) Build the manifests
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Fine-tune the pretrained codec with the LACE wrapper
python run.py --stages train --training_config conf/training.yaml

# 3) Resynthesize the test set at the compression rate set in the config, then score it
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml

# 4) Pack a self-contained bundle, then upload it
python run.py --stages pack_model upload_model \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --publication_config conf/publication.yaml

# 5) Build the Gradio demo and push it as a Space
python run.py --stages pack_demo upload_demo \
    --training_config conf/training.yaml \
    --demo_config conf/demo.yaml
```

There is no `collect_stats` stage to run: the chunk-based data loader needs no shape files.

Training writes the wrapped model spec to `exp/${exp_tag}/lace_model.yaml`.
`conf/inference.yaml` rebuilds the model from that spec and loads `exp/${exp_tag}/last.ckpt` on top (`espnet3.systems.lace.src.inference.LACEAudioCoding`); the template's inference contract takes the test set's `audio` and writes the resynthesized waveforms as WAV files listed in `exp/${exp_tag}/inference/test/resyn_audio.scp`.
`measure` scores them against the test set's own waveforms (`dataset:wav_path` in `conf/metrics.yaml`) with versa.
Set `model.rate` in `conf/inference.yaml` to choose the compression rate, and `model.anchor_start_layer` to make later layers reuse the boundaries of earlier ones.
To sweep rates, copy the config once per rate (for example `conf/inference_r025.yaml`): the output directory is named after the config file, so the runs do not overwrite each other.

`model.compression_model.name` in `conf/training.yaml` selects how frames are merged:

| Name | Boundary rule | Meaning of `rate` |
|---|---|---|
| `cosine_similarity` | Start a new segment when the cosine similarity to the previous frame drops below a threshold. | The similarity threshold. |
| `density_peak` | VARSTok: grow clusters around density-peak seed frames, up to `max_span` frames, while neighbours stay similar enough. | The similarity threshold. |
| `dp_segmentation` | CodecSlime: dynamic programming for the segmentation with the least L2 distortion. | The segment fraction, `round(rate * T)` segments. |
| `identity` | No merging; the pretrained codec with the wrapper in place, as a no-compression reference. | Ignored. |

For the first two models a higher rate therefore means more segments and less compression, and `dp_segmentation` follows the same direction by construction.
During training a rate is sampled per forward pass in `[min_rate, max_rate]`; validation and inference use a fixed rate.

The bundle `pack_model` writes carries `lace_model.yaml` and the fine-tuned checkpoint; the pretrained codec it names by Hugging Face id is downloaded again when the bundle is loaded.
Nothing in the bundle imports recipe code: the packed `conf/inference.yaml` names only installed classes and declares its contract inline, so the bundle loads with `trust_user_code=False`.
The demo stages require `--training_config` because the demo config resolves `${exp_tag}` from it.
`conf/demo.yaml` points the demo at `${upload_demo.hf_repo}`, so `upload_model` has to publish to that repo before the Space can load a model.

Outside the recipe, a packed or published model loads through `espnet3.publication.InferenceModel`, and any `LACEAudioCoding` argument replaces the packed one:

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_packed("exp/train_encodec_lace_libritts/model_pack", rate=0.25)
resynthesized = model("utt.wav")["resyn_audio"]  # an Audio at 24 kHz
```

A published bundle loads the same way by its Hugging Face id, with `InferenceModel.from_pretrained(tag, rate=...)`.

## Results

### EnCodec with LACE

- Training config: `conf/training.yaml`
- Inference config: `conf/inference.yaml`
- Metrics config: `conf/metrics.yaml`

`conf/training.yaml` fine-tunes the LibriTTS EnCodec with the `cosine_similarity` compression model, and `conf/metrics.yaml` scores the test-set resynthesis with the VERSA `signal_metric` and `pseudo_mos` groups.
No evaluation results are available for this configuration yet.

#### Pretrained model

No pretrained model is available for this configuration yet.

### DAC with LACE (DP segmentation)

- Training config: `conf/training_lace_dac.yaml`
- Inference config: `conf/inference.yaml`
- Metrics config: `conf/metrics.yaml`

The DAC model of the LACE paper: the LibriTTS DAC of the ESPnet model zoo ([`espnet/libritts_dac_24k`](https://huggingface.co/espnet/libritts_dac_24k)) fine-tuned with the `dp_segmentation` compression model (`max_span: 4`), rates sampled in `[0.25, 1.0]` per quantizer, for 30k updates on 4 GPUs.
The checkpoint was trained with the paper's own code and converted to the checkpoint layout of this recipe.
`conf/training_lace_dac.yaml` is the same configuration written for this recipe.
It is scored on test-clean (4,837 utterances) at rate 0.5 by the `measure` stage with VERSA `signal_metric` and `pseudo_mos`.

| dataset | sdr | si_snr | ci_sdr | utmos | dns_overall | dns_p808 | plcmos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test-clean | 9.13 | 7.29 | 9.04 | 3.96 | 3.11 | 3.67 | 4.04 |

#### Pretrained model

- Model: [`NewGame/libritts_lace_dac`](https://huggingface.co/NewGame/libritts_lace_dac)

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained("NewGame/libritts_lace_dac", rate=0.5)
resynthesized = model("utt.wav")["resyn_audio"]  # an Audio at 24 kHz
```

### EnCodec with LACE (DP segmentation)

- Training config: `conf/training_lace_encodec.yaml`
- Inference config: `conf/inference.yaml`
- Metrics config: `conf/metrics.yaml`

The EnCodec model of the LACE paper: the LibriTTS EnCodec of the ESPnet model zoo ([`espnet/libritts_encodec_24k`](https://huggingface.co/espnet/libritts_encodec_24k)) fine-tuned with the `dp_segmentation` compression model (`max_span: 4`), rates sampled in `[0.25, 1.0]` per quantizer, for 30k updates on 4 GPUs.
The checkpoint was trained with the paper's own code and converted to the checkpoint layout of this recipe.
`conf/training_lace_encodec.yaml` is the same configuration written for this recipe.
It is scored on test-clean (4,837 utterances) at rate 0.5 by the `measure` stage with VERSA `signal_metric` and `pseudo_mos`.

| dataset | sdr | si_snr | ci_sdr | utmos | dns_overall | dns_p808 | plcmos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test-clean | 4.36 | 1.45 | 4.31 | 3.95 | 3.13 | 3.77 | 4.28 |

#### Pretrained model

- Model: [`NewGame/libritts_lace_encodec`](https://huggingface.co/NewGame/libritts_lace_encodec)

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained("NewGame/libritts_lace_encodec", rate=0.5)
resynthesized = model("utt.wav")["resyn_audio"]  # an Audio at 24 kHz
```

### SoundStream with LACE (DP segmentation)

- Training config: `conf/training_lace_soundstream.yaml`
- Inference config: `conf/inference.yaml`
- Metrics config: `conf/metrics.yaml`

The SoundStream model of the LACE paper: the LibriTTS SoundStream of the ESPnet model zoo ([`espnet/libritts_soundstream24k`](https://huggingface.co/espnet/libritts_soundstream24k)) fine-tuned with the `dp_segmentation` compression model (`max_span: 4`), rates sampled in `[0.25, 1.0]` per quantizer, for 30k updates on 4 GPUs.
The checkpoint was trained with the paper's own code and converted to the checkpoint layout of this recipe.
`conf/training_lace_soundstream.yaml` is the same configuration written for this recipe.
It is scored on test-clean (4,837 utterances) at rate 0.5 by the `measure` stage with VERSA `signal_metric` and `pseudo_mos`.

| dataset | sdr | si_snr | ci_sdr | utmos | dns_overall | dns_p808 | plcmos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test-clean | 3.12 | 0.03 | 3.07 | 3.73 | 3.09 | 3.74 | 4.19 |

#### Pretrained model

- Model: [`NewGame/libritts_lace_soundstream`](https://huggingface.co/NewGame/libritts_lace_soundstream)

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained("NewGame/libritts_lace_soundstream", rate=0.5)
resynthesized = model("utt.wav")["resyn_audio"]  # an Audio at 24 kHz
```

The per-codec configs expect four GPUs (`num_device: 4`).
For another setup, change `num_device` and `dataloader.train.iter_factory.batch_size` together.

## Citation

```bibtex
@inproceedings{trachu2026lace,
  title     = {{LACE}: Layer-Wise Compression for Dynamic Frame Rate Codecs},
  author    = {Trachu, Thanapat and Cornell, Samuele and Chen, William and Watanabe, Shinji},
  booktitle = {Proc. IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2609.17509}
}
```
