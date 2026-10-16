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

The `measure` stage runs [versa](https://github.com/wavlab-speech/versa), which
has to be installed in the same environment that runs the stage. Install it
with `tools/installers/install_versa.sh`. The pseudo-MOS metrics also need
`speechmos` and `onnxruntime`. When those are missing, versa logs a load
failure, skips that whole metric group and still exits 0. Read the scores in
`avg_result.json` to see what was actually computed rather than trusting the
exit status.

### Choosing the checkpoint

`conf/inference.yaml` and `conf/publication.yaml` both name the checkpoint to
use, and they ship pointing at the checkpoint of the published run. After
training your own model, set both of them to a file your run wrote under
`exp/${exp_tag}/`.

```yaml
# conf/inference.yaml
model:
  model_file: ${exp_dir}/valid.mel_loss.ave_5best.pth

# conf/publication.yaml
pack_model:
  files:
    model_file: ${exp_dir}/valid.mel_loss.ave_5best.pth
```

The two files must name the same checkpoint. Entries under `pack_model.files`
are copied into the bundle even when the `exclude` list would otherwise drop
them, and the packed inference config is rewritten to point at that bundled
copy. If `inference.yaml` names a checkpoint that `publication.yaml` does not
register, the bundle references a file it does not contain and
`InferenceModel.from_packed` fails.

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
the Space can load a model.

## Pretrained model and demo

Trained on LibriTTS with this recipe's EnCodec configuration and published from
its `epoch95_step1201215` checkpoint.

- Model: [NewGame/libritts_codec_train_encodec_libritts](https://huggingface.co/NewGame/libritts_codec_train_encodec_libritts)
- Demo Space: [NewGame/libritts_codec_train_encodec_libritts](https://huggingface.co/spaces/NewGame/libritts_codec_train_encodec_libritts)

The Space runs on ZeroGPU. It takes an uploaded file or a microphone recording
and resynthesizes it through the codec.

The published model also loads directly.

```python
from espnet3.publication.inference_model import InferenceModel

model = InferenceModel.from_pretrained(
    "NewGame/libritts_codec_train_encodec_libritts", trust_user_code=True
)
```

`trust_user_code=True` is required because the bundle carries this recipe's
`src` package, which formats the model output.
