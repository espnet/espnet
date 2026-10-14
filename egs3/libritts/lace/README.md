# LibriTTS LACE recipe

LACE (Layer-Adaptive Codec Encoding) fine-tuning of a neural codec on LibriTTS, the ESPnet3 recipe of [LACE: Layer-Wise Compression for Dynamic Frame Rate Codecs](https://arxiv.org/abs/2609.17509) (SLT 2026).
LACE wraps the residual vector quantizer of a trained codec so that, before every codebook layer, similar neighbouring frames are merged into segments and each layer runs at its own, lower frame rate.
The compression `rate` controls how much is merged: a higher rate keeps more segments and compresses less.
The model code lives in `espnet3/systems/lace`; the default configuration fine-tunes the LibriTTS EnCodec published from the ESPnet3 codec recipe, [`NewGame/libritts_codec_train_encodec_libritts`](https://huggingface.co/NewGame/libritts_codec_train_encodec_libritts).

## Corpus

On the first run `create_dataset` downloads the five LibriTTS subsets into `downloads/LibriTTS`, roughly 80 GB, and writes the `train`/`valid`/`test` manifests under `data/manifest`.

If LibriTTS is already on disk, point the recipe at it rather than downloading a second copy: set `builder.dataset_path` in `dataset/config.yaml` to the directory that holds the `LibriTTS` folder.
An absolute path is used exactly as written, and a relative one is resolved from the recipe directory.
A subset counts as prepared only once it carries a `.complete` marker, which `create_dataset` writes after extracting that subset; a corpus staged any other way needs the markers created by hand (`touch <dataset_path>/LibriTTS/<subset>/.complete`), or the recipe downloads the subsets again.

## Baseline codec

The baseline is a published ESPnet3 model bundle (the `pack_model` output of the codec recipe, uploaded by its `upload_model` stage).
`model.pretrained_model_tag` names it by its Hugging Face id, and `espnet_model_zoo` downloads and caches it on first use.
An espnet2 zoo model (`meta.yaml` with `train_config`) is accepted the same way, as long as its checkpoint matches the current espnet2 code.
To fine-tune a codec you trained with an ESPnet3 codec recipe instead, run its `pack_model` stage and set `model.pretrained_model_dir` to the resulting `model_pack` directory (or `pretrained_model_tag` to the repo its `upload_model` stage pushed to).

## Quick start

```bash
# 1) Build the manifests
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Fine-tune the published baseline with the LACE wrapper
python run.py --stages train --training_config conf/training.yaml

# 3) Resynthesize the test set at the compression rate set in the config, then score it
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

There is no `collect_stats` stage to run: the chunk-based data loader needs no shape files.

Training writes the wrapped model spec to `exp/${exp_tag}/lace_model.yaml`.
`conf/inference.yaml` rebuilds the model from that spec and loads `exp/${exp_tag}/last.ckpt` on top, through the system's `Inference` (`espnet3.systems.lace.inference.Inference`), which takes the test set's `audio` and writes the resynthesized waveforms as WAV files listed in `exp/${exp_tag}/inference/test/resyn_audio.scp`.
`measure` scores them against the test set's own waveforms (`dataset:wav_path` in `conf/metrics.yaml`) with versa.
Set `model.rate` in `conf/inference.yaml` to choose the compression rate, and `model.anchor_start_layer` to make later layers reuse the boundaries of earlier ones.
To sweep rates, copy the config once per rate (for example `conf/inference_r025.yaml`): the output directory is named after the config file, so the runs do not overwrite each other.

Outside the recipe, a packed or published model loads through the inference API, and any `LACEAudioCoding` argument replaces the packed one:

```python
from espnet3.api.inference import load

model = load("exp/train_encodec_lace_libritts/model_pack", rate=0.25)
resynthesized = model("utt.wav")["resyn_audio"]  # an Audio at 24 kHz
```

## Publishing

```bash
# Pack a self-contained bundle, then upload it
python run.py --stages pack_model upload_model \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --publication_config conf/publication.yaml

# Build the Gradio demo and push it as a Space
python run.py --stages pack_demo upload_demo \
    --training_config conf/training.yaml \
    --demo_config conf/demo.yaml
```

The bundle carries `lace_model.yaml` and the fine-tuned checkpoint; the baseline it names by Hugging Face id is downloaded again when the bundle is loaded.
`meta.yaml` records the system, so `espnet3.api.inference.load` finds the `Inference` class without being told.
The demo stages require `--training_config` because the demo config resolves `${exp_tag}` from it.
`conf/demo.yaml` points the demo at `${upload_demo.hf_repo}`, so `upload_model` has to publish to that repo before the Space can load a model.

## Compression models

`model.compression_model.name` in `conf/training.yaml` selects how frames are merged:

| Name | Boundary rule | Meaning of `rate` |
|---|---|---|
| `cosine_similarity` | Start a new segment when the cosine similarity to the previous frame drops below a threshold. | The similarity threshold. |
| `density_peak` | VARSTok: grow clusters around density-peak seed frames, up to `max_span` frames, while neighbours stay similar enough. | The similarity threshold. |
| `dp_segmentation` | CodecSlime: dynamic programming for the segmentation with the least L2 distortion. | The segment fraction, `round(rate * T)` segments. |
| `identity` | No merging; the baseline codec with the wrapper in place. | Ignored. |

For the first two models a higher rate therefore means more segments and less compression, and `dp_segmentation` follows the same direction by construction.
During training a rate is sampled per forward pass in `[min_rate, max_rate]`; validation and inference use a fixed rate.

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
