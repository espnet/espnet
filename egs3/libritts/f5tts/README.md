# ESPnet3 LibriTTS F5-TTS recipe

This recipe trains, evaluates and publishes **F5-TTS**, a flow-matching non-autoregressive TTS model with zero-shot voice cloning from a reference utterance, on LibriTTS.
It runs `espnet3.systems.f5tts.system.F5TTSSystem` through the shared runner in `egs3/TEMPLATE/f5tts/run.py`; every config in `conf/` is merged over the template default of the same name.

## 1. Prepare data and train

```bash
# Download the corpora and build every manifest (run once)
python run.py --stages create_dataset --training_config conf/training_small.yaml

# Filter utterances by duration
python run.py --stages remove_long_short --training_config conf/training_small.yaml

# Build the token list
python run.py --stages create_token_list --training_config conf/training_small.yaml

# Collect feature statistics
python run.py --stages collect_stats --training_config conf/training_small.yaml

# Train
python run.py --stages train --training_config conf/training_small.yaml
```

`conf/training_small.yaml` is the recipe's only training config: the F5TTS_Small architecture (hidden size 768, depth 18, 12 attention heads), targeting the LibriTTS rows of arXiv 2410.06885 Table 9.
It is complete on its own, because the `infer` stage and a packed model rebuild the model and the tokenizer from it alone.
Training logs go to TensorBoard under `exp/training_small/tensorboard`.

`create_dataset` prepares both the training data and the eval data.
It downloads:

- **LibriTTS** (OpenSLR 60), the five subsets listed in `dataset/config.yaml`, and writes `data/manifest/{train,valid,test}.tsv`.
- **LibriSpeech `test-clean`** (OpenSLR **12**, a different corpus) plus the cross-sentence pair list from the F5-TTS repo, and writes `data/librispeech_pc/manifest.tsv`, the eval manifest that `conf/inference.yaml` reads.

Everything is idempotent: extracted subsets carry a `.complete` marker and the pair list is skipped when present, so re-running `create_dataset` transfers nothing.

## 2. Synthesize

```bash
python run.py --stages infer \
    --training_config conf/training_small.yaml \
    --inference_config conf/inference.yaml
```

`--training_config` is required here.
`conf/inference.yaml` leaves `exp_tag` empty so it inherits experiment identity from the training config, and `run.py` rejects an inference config with no experiment identity of its own.

`conf/inference.yaml` runs the paper protocol: LibriSpeech-PC test-clean cross-sentence, 1127 same-speaker prompt/target pairs.
To evaluate in-domain on the LibriTTS `valid`/`test` splits with cross-speaker prompts instead, swap in `conf/inference_libritts.yaml`.
Both inference configs pin the same `conf/training_small.yaml`, so either one loads a checkpoint from Section 1 without further edits.

If you add a training config for a different architecture, note that `--training_config` only propagates `exp_tag` and `exp_dir` into the inference config (`espnet3/utils/run_utils.py`'s `_TRAINING_CONTEXT_KEYS`); it never overrides `model.train_config`.
Point the inference config's own `model.train_config` at the matching training config, or the checkpoint will fail to load with a shape mismatch.

## 3. Score

```bash
python run.py --stages measure \
    --training_config conf/training_small.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

Scoring runs through VERSA and reports WER, speaker similarity, and UTMOS.

### Dependencies

`conf/metrics.yaml` needs four packages, and espnet's `tools/installers/install_versa.sh` installs only the first: it takes VERSA's `[audio]` extra and nothing else.

| Package | Needed by | Install |
|---|---|---|
| `versa` | the stage itself | `tools/installers/install_versa.sh` |
| `faster-whisper` | `fwhisper_wer` (the ASR pass) | `tools/install_fwhisper.sh` inside `tools/versa` |
| `openai-whisper` | `fwhisper_wer`'s `text_cleaner: whisper_basic` | `pip install openai-whisper` |
| `s3prl` | `speaker`, for the WavLM front-end of `espnet/voxcelebs12_ecapa_wavlm_joint` | `pip install s3prl` |

`openai-whisper` is a **different package** from `faster-whisper`: the cleaner is routed through `espnet2/text/cleaner.py`, which only enables its `whisper_*` branches when OpenAI's `whisper` imports.

Install all four before running `measure`.
The stage fails when a configured metric yields no value for any utterance, so a missing package is reported rather than silently dropped from the summary.

`conf/metrics.yaml` documents how each metric maps onto the official F5-TTS scorer, including the one metric that cannot be matched exactly.
In short: WER and UTMOS are equivalent to the official implementations, but speaker similarity uses an ESPnet-SPK model rather than the official UniSpeech checkpoint, so SIM values are comparable across your own checkpoints but not against the numbers published in the paper.

## 4. Publish the model

```bash
python run.py --stages pack_model \
    --training_config conf/training_small.yaml \
    --inference_config conf/inference.yaml \
    --publication_config conf/publication.yaml

python run.py --stages upload_model \
    --training_config conf/training_small.yaml \
    --publication_config conf/publication.yaml
```

`pack_model` writes a self-contained bundle to `exp/training_small/model_pack`: `last.ckpt`, the training config the model is rebuilt from, the token list under `data/tokens`, the recipe's `src/` and `dataset/` code, and a README rendered from `measure`'s results when they exist.
`conf/publication.yaml` only overrides the template's `pack_model.include`, because this recipe keeps its token list in `data/tokens`.

The bundle loads back through the inference API without naming the recipe:

```python
from espnet3.api.inference import load

model = load("exp/training_small/model_pack", device="cuda:0")
output = model("Hello world.", "prompt.wav", "The transcript of prompt.wav.")
output["wav"].rate, output["wav"].array.shape
```

`upload_model` pushes the bundle to `espnet/libritts_f5tts_training_small` on the Hugging Face Hub (run `hf auth login` first).

## 5. Pack and upload a demo

```bash
python run.py --stages pack_demo \
    --training_config conf/training_small.yaml --demo_config conf/demo.yaml

python run.py --stages upload_demo \
    --training_config conf/training_small.yaml --demo_config conf/demo.yaml
```

`pack_demo` copies `src/app.py`, the Gradio launcher, together with the demo config and a Space README into `demo/`, pointing it at the bundle from Section 4.
The app takes the text to synthesize, a reference recording and its transcript, and returns the synthesized speech.
Run it locally with `python demo/app.py`, or push it to a Space with `upload_demo`.
