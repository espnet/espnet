# ESPnet3 LACE recipe

Fine-tunes a published espnet2 GAN codec with the LACE (Layer-Adaptive Codec Encoding) quantizer wrapper, so every codebook layer runs at its own frame rate.
The model is described in "LACE: Layer-Wise Compression for Dynamic Frame Rate Codecs" (Trachu, Cornell, Chen, Watanabe, SLT 2026, https://arxiv.org/abs/2609.17509).

## Quick start

```bash
# 0) Edit configs to set paths and pick the pretrained codec.
#    `conf/training.yaml:model.pretrained_model_tag` names the published
#    codec bundle to fine-tune (or `model.pretrained_model_dir` for a local
#    `pack_model` output); `model.compression_model` picks the segmentation.
#    The training config has no `task:` key on purpose: the model is a
#    Hydra target (`espnet3.systems.lace.src.lace.LACE`), and `train`
#    writes its spec to `${exp_dir}/lace_model.yaml` for `infer` to rebuild.
#    When `--training_config` is also passed to `infer` or `measure`, run.py
#    propagates experiment path fields from training into inference/metrics.

# 1) Build the manifests (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Fine-tune the pretrained codec with the LACE wrapper (no collect_stats needed:
#    the chunk-based loader uses no shape files)
python run.py --stages train --training_config conf/training.yaml

# 3) Resynthesize the test set at `model.rate` of conf/inference.yaml
#    (one WAV per utterance, listed in resyn_audio.scp)
python run.py --stages infer --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 4) Score the resynthesis against the test set's own waveforms with versa
python run.py --stages measure --training_config conf/training.yaml \
    --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml

# 5) Pack a self-contained bundle / a Gradio demo (upload stages push them)
python run.py --stages pack_model --training_config conf/training.yaml \
    --inference_config conf/inference.yaml --publication_config conf/publication.yaml
python run.py --stages pack_demo --training_config conf/training.yaml \
    --demo_config conf/demo.yaml
```

`measure` needs `versa` (`tools/installers/install_versa.sh`).
A recipe's `metrics.yaml` lists the scores its configuration promises under `outputs`, and the stage fails when VERSA drops one instead of reporting a partial table.

## Using a packed model

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained("<hf_repo>")
audio = model("speech.wav")["resyn_audio"]  # an Audio: audio.array (float32) at audio.rate

# Any LACEAudioCoding argument replaces the packed one, which is how a rate sweep reads:
model = InferenceModel.from_pretrained("<hf_repo>", rate=0.25)
```

The input may be a path, a `(rate, samples)` pair or an array; it is resampled to the codec's rate and averaged to one channel before encoding, as declared in `conf/inference.yaml`'s `contract`.
See `egs3/libritts/lace/README.md` for a complete recipe.
