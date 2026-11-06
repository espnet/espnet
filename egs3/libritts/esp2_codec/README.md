# LibriTTS codec recipe

Neural codec training on LibriTTS, ported from
[`egs2/libritts/codec1`](../../../egs2/libritts/codec1) to ESPnet3.
The default configuration trains EnCodec at 24 kHz.

## Corpus

On the first run `create_dataset` downloads the five LibriTTS subsets into
`downloads/LibriTTS`, roughly 80 GB.

If LibriTTS is already on disk, point the recipe at it rather than downloading
a second copy. Set `builder.dataset_path` in `dataset/config.yaml` to the
directory that holds the `LibriTTS` folder. An absolute path is used exactly as
written, and a relative one is resolved from the recipe directory.

```yaml
# dataset/config.yaml
builder:
  dataset_path: /corpora/libritts
```

A subset counts as prepared only once it carries a `.complete` marker, which
`create_dataset` writes after extracting that subset. A corpus staged any other
way has no markers, and the recipe will download the subsets again. Create the
markers yourself to prevent that.

```bash
for subset in train-clean-100 train-clean-360 train-other-500 dev-clean test-clean; do
    touch /corpora/libritts/LibriTTS/${subset}/.complete
done
```

## Quick start

```bash
# 1) Build the manifests
python run.py --stages create_dataset \
    --training_config conf/training_encodec.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats \
    --training_config conf/training_encodec.yaml

# 3) Train
python run.py --stages train \
    --training_config conf/training_encodec.yaml

# 4) Resynthesize the test set and score it
python run.py --stages infer measure \
    --training_config conf/training_encodec.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`infer` writes one resynthesized WAV per test utterance under
`exp/${exp_tag}/inference/test/` and lists them in `wav.scp`. `measure` reads
the ground-truth file of each utterance from the test set itself (the
`wav_path` column, bound as `dataset:wav_path` in `conf/metrics.yaml`), so
nothing but the model's output is written by `infer`.

The `measure` stage runs [versa](https://github.com/wavlab-speech/versa), which
has to be installed in the same environment that runs the stage. Install it
with `tools/installers/install_versa.sh`. The pseudo-MOS metrics also need
`speechmos` and `onnxruntime`. When those are missing, versa logs a load
failure, skips that whole metric group and still exits 0; `conf/metrics.yaml`
therefore lists the scores the configuration promises under `outputs`, and
`measure` fails when one of them is missing instead of reporting a partial
table.

### Choosing the checkpoint

Inference and publication read `exp/${exp_tag}/last.ckpt`, the link the
trainer keeps to the final checkpoint of the run, so a freshly trained model
infers and packs without any path change. To use another checkpoint, set
`model.model_file` in `conf/inference.yaml` and `pack_model.files.model_file`
in `conf/publication.yaml` to the same file.

## Publishing

```bash
# Pack a self-contained bundle, then upload it
python run.py --stages pack_model upload_model \
    --training_config conf/training_encodec.yaml \
    --inference_config conf/inference.yaml \
    --publication_config conf/publication.yaml

# Build the Gradio demo and push it as a Space
python run.py --stages pack_demo upload_demo \
    --training_config conf/training_encodec.yaml \
    --demo_config conf/demo.yaml
```

The demo stages require `--training_config` because the demo config resolves
`${exp_tag}` from it. `conf/demo.yaml` points the demo at
`${upload_demo.hf_repo}`, so `upload_model` has to publish to that repo before
the Space can load a model. The demo's input and output widgets come from the
inference contract in `conf/inference.yaml`.

## Results

EnCodec trained with `conf/training_encodec.yaml` on train-clean-100,
train-clean-360 and train-other-500 (4x V100), scored on test-clean (4,837
utterances) by the `measure` stage with VERSA `signal_metric` and
`pseudo_mos`.

| sdr | si_snr | ci_sdr | utmos | dns_overall | dns_p808 | plcmos |
| --- | --- | --- | --- | --- | --- | --- |
| 6.05 | 3.47 | 5.99 | 3.99 | 3.12 | 3.77 | 4.32 |

## Pretrained model and demo

The model above, published from the last checkpoint of its run with the
`pack_model` and `pack_demo` stages.

- Model: [NewGame/libritts_codec_train_encodec_libritts](https://huggingface.co/NewGame/libritts_codec_train_encodec_libritts)
- Demo Space: [NewGame/libritts_codec_train_encodec_libritts](https://huggingface.co/spaces/NewGame/libritts_codec_train_encodec_libritts)

The Space runs on ZeroGPU. It takes an uploaded file or a microphone recording
and resynthesizes it through the codec.

The published model also loads directly. The input is anything that carries
audio (a path, a `(rate, samples)` pair, an array at 24 kHz); the output `wav`
is an `Audio` with `.array` and `.rate`.

```python
from espnet3.publication.inference_model import InferenceModel

model = InferenceModel.from_pretrained("NewGame/libritts_codec_train_encodec_libritts")
resynthesized = model("utt.wav")["wav"]
```
