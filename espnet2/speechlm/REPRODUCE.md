# Bagpiper: public-checkpoint reproduction

This walkthrough starts with a fresh environment and public model/data downloads.
It covers native audio understanding and generation, full-parameter fine-tuning
on eight GPUs, checkpoint recovery/export, and the ESPnet vLLM server. The
[experiment log](EXPERIMENT_LOG.md) records the actual failures, source fixes,
commands, timings, and results from the 2026-09-09 run.

This is a **smoke experiment**, not a reproduction of the paper's benchmark
scores. It deliberately trains on a subset of LibriSpeech **test-clean**. Never
report results on those examples as held-out performance. The preparation helper
is under `test_utils/`; the recipes themselves still take prepared data only.

## 1. Workspace and installation

Use an ESPnet checkout containing the fixes accompanying this guide. The tested
branch is `bagpiper-e2e-validation`, based on upstream commit
`4fb9218b9b6427e3c4dc1582958facbf17298e4e`; installation PR #6645 and trainer/recipe
PRs #6646/#6647 are already included in that baseline.
The source-fix commit is `6239adaf90`.

The run used Linux, eight H100 80 GB GPUs, Python 3.12, and a local CUDA 12.9
toolkit to build FlashAttention-3. Other GPUs need a supported attention backend.
The two recipe YAMLs select it independently for the language model and audio
encoder. Allow space for model downloads and checkpoints: each full FP32
model/Adam DCP in this experiment is about 96 GiB; a BF16 model export is about
17 GiB. CPU model construction/export also needs RAM for full model tensors.

From the ESPnet repository root:

```bash
conda create -n bagpiper-repro python=3.12 pip -y
conda activate bagpiper-repro
set -euo pipefail
export BAGPIPER_REPO="$PWD"
export BAGPIPER_RUN="$PWD/.artifacts/bagpiper-repro"
export BAGPIPER_PYTHON="$(command -v python)"
mkdir -p "$BAGPIPER_RUN"/{models,data,configs,results,logs,source}
export HF_HOME="$BAGPIPER_RUN/hf"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[speechlm]'
pip install setuptools wheel packaging ninja psutil
pip install --no-build-isolation 'causal-conv1d>=1.6.2'
```

For H100/H800, build FA3 with the CUDA toolkit on `PATH` and `CUDA_HOME` pointing
to its installation. See the shorter [installation guide](INSTALL.md) for
platform-dependent choices. These source/version values record the tested run;
they are not a universal dependency lock for every GPU.

```bash
git clone https://github.com/Dao-AILab/flash-attention.git "$BAGPIPER_RUN/source/flash-attention"
git -C "$BAGPIPER_RUN/source/flash-attention" checkout 9d61d35ba876834539af7df369fbabef1ad1e5d7
MAX_JOBS=32 NVCC_THREADS=2 pip install --no-build-isolation \
    "$BAGPIPER_RUN/source/flash-attention/hopper"
pip check
pip freeze > "$BAGPIPER_RUN/logs/training-environment.txt"
```

The full FA3 build took 23.4 minutes on this host. Choose build parallelism for
your CPU/RAM. PyTorch wheels provide CUDA runtime libraries; compiling these
extensions additionally requires a local toolkit/compiler. If pip inherits a
broken extra package index, inspect that configuration; in this run the command
was retried with `env -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null` without
changing global pip configuration.

| Component | Training environment | Separate vLLM environment |
| --- | --- | --- |
| Python | 3.12.14 | 3.12 |
| PyTorch | 2.11.0+cu128 | 2.13.0+cu132 |
| Transformers | 5.14.1 | 5.16.1 |
| TorchTitan | 0.2.2 | Not used |
| Liger | 0.8.2 | Not used by this model |
| FlashAttention | FA3 3.0.0, built above | Fork's precompiled installation |

## 2. Download the released checkpoints

The checkpoints come from the [ESPnet Bagpiper collection](https://huggingface.co/collections/espnet/bagpiper).
Base is used for audio ↔ caption and general fine-tuning. TTS-SFT is used for the
instruction dialogue recipe and vLLM's think/describe/render client.

```bash
hf download espnet/bagpiper base.pt train_stage2_qwen3_base.yaml \
    --revision cde214717b75190c0b117c0f3ad59b1412e3c695 \
    --local-dir "$BAGPIPER_RUN/models/bagpiper"
hf download espnet/bagpiper-tts-sft model.pt train_bagpiper_tts.yaml inference.yaml \
    --revision 675e2fafccc7dd7205fad6f8fdc4451f9ee6f768 \
    --local-dir "$BAGPIPER_RUN/models/bagpiper-tts-sft"
sha256sum "$BAGPIPER_RUN/models/bagpiper/base.pt" \
    "$BAGPIPER_RUN/models/bagpiper-tts-sft/model.pt"
```

Expected SHA-256 values:

```text
c9917c1237bd44ea0ea8b41746e5ec5aecae92273538c9b6e4040d26f69a6521  base.pt
168fe0aca32f4b1636e4ee2e63ef3880427be352bd6369a1f5ba3b49750f91da  model.pt
```

Native checkpoints already contain the language model, audio tower, and codec.
Model construction first loads the pretrained components, then strictly loads
the released weights over them. Pre-cache the public configuration/tokenizer
assets and pretrained component weights before enabling offline mode below.
The following revisions identify the tested tokenizer/config files. Cache their
`main` references as well, because the recipe resolves these model names offline.
If `main` has moved, the check stops instead of silently testing different assets:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
sources = {
    "Qwen/Qwen3-8B-Base": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": "26291f793822fb6be9555850f06dfe95f2d7e695",
    "hf-audio/xcodec-hubert-general": "240735a1ade315cb9b22604fe898e7df03d67378",
}
for repo, revision in sources.items():
    path = snapshot_download(repo,
        allow_patterns=["*.json", "*.txt", "*.model", "*.jinja",
                        "*.safetensors", "*.bin"])
    assert Path(path).name == revision, (repo, path, revision)
    print(repo, path)
PY
```

If the revision check fails, upstream `main` has changed. Compare the new
configuration/tokenizer assets with the recorded revisions before validating
that new combination.

## 3. Tests and LibriSpeech inputs

Run the SpeechLM CPU tests and the opt-in tests that use real CUDA kernels:

```bash
pip install pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -o addopts= -q test/espnet2/speechlm
CUDA_VISIBLE_DEVICES=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -o addopts= -q \
    test_utils/speechlm/test_fused_loss_cuda.py \
    test_utils/speechlm/test_native_moe_cuda.py
```

The CUDA tests compare forward values and gradients, and check that native
grouped MM is actually called. The released Qwen3-8B checkpoint itself is dense;
the MoE check therefore uses a small, locally initialized Qwen3-MoE model.

Download the official corpus and select a deterministic subset:

```bash
curl --fail --location --retry 3 --continue-at - \
    https://www.openslr.org/resources/12/test-clean.tar.gz \
    --output "$BAGPIPER_RUN/data/test-clean.tar.gz"
tar --no-same-owner -xzf "$BAGPIPER_RUN/data/test-clean.tar.gz" -C "$BAGPIPER_RUN/data"
python test_utils/speechlm/prepare_bagpiper_smoke.py \
    --librispeech "$BAGPIPER_RUN/data/LibriSpeech/test-clean" \
    --output-dir "$BAGPIPER_RUN/data/smoke"
```

The helper chooses clips lasting 2–10 seconds by round-robin speaker order:
96 clips, 40 speakers, 543.355 seconds. The first 64 IDs form training and the
last 32 validation. It writes Lhotse audio metadata, transcripts, sample IDs,
and the dataset JSONs. Paths are absolute, so retain the data location.

Create inference and short training configurations from the existing recipes:

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml
run = Path(os.environ["BAGPIPER_RUN"])
config = yaml.safe_load(Path("egs2/bagpiper/speechlm1/conf/tuning/train_sft.yaml").read_text())
config["data_loading"].update(batch_size=2048, num_workers=0)
config["trainer"].update(max_step=3, save_interval=2, gradient_accumulation_steps=2)
config["trainer"]["lr_scheduler"].update(warmup_steps=1, decay_end_step=6)
config["trainer"]["titan_config"]["dp_shard"] = 8
(run / "configs/smoke-train.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
config["trainer"]["max_step"] = 5
(run / "configs/smoke-resume.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
config["data_loading"].update(batchfy_method="bucket", batch_size=-1, num_workers=0)
(run / "configs/inference_model.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
for name, modality, settings in [
    ("caption", "text", dict(temperature=0.6, topk=20, cfg=1, max_step=1024, min_step=1)),
    ("audio", "audio", dict(temperature=0.8, topk=20, cfg=3, max_step=800, min_step=20)),
]:
    infer = dict(dtype="bfloat16", enforce_modality=[modality])
    infer[modality] = settings
    (run / f"configs/{name}.yaml").write_text(yaml.safe_dump(infer))
PY
```

Vocabulary IDs are fixed by the source: special tokens first, then text, then
other discrete modalities by name. YAML key order must not change them. For the
released Qwen3/Xcodec checkpoint, text occupies `[256, 152192)` and audio starts
at `152192`. The validation found and fixed an older order-dependent layout that
silently corrupted restored training after a default `yaml.safe_dump`.

## 4. Audio → rich caption → audio

For one GPU, omit `--rank` and `--world-size`. To reproduce the six-GPU caption
run, launch one process per selected GPU and wait for all of them:

```bash
export HF_HUB_OFFLINE=1
pids=()
for gpu in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES="$gpu" python -m espnet2.speechlm.bin.inference \
        --rank "$((gpu + 1))" --world-size 6 \
        --train-config "$BAGPIPER_RUN/configs/inference_model.yaml" \
        --inference-config "$BAGPIPER_RUN/configs/caption.yaml" \
        --model-checkpoint "$BAGPIPER_RUN/models/bagpiper/base.pt" \
        --output-dir "$BAGPIPER_RUN/results/captions" \
        --test-unregistered-specifier "audio_to_text:clean:$BAGPIPER_RUN/data/smoke/all.json" \
        > "$BAGPIPER_RUN/logs/caption-gpu$gpu.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
python test_utils/speechlm/prepare_bagpiper_smoke.py \
    --librispeech "$BAGPIPER_RUN/data/LibriSpeech/test-clean" \
    --output-dir "$BAGPIPER_RUN/data/smoke" \
    --captions "$BAGPIPER_RUN/results/captions"
```

Each GPU rank is **one-based** on the CLI; `--num-workers` defaults to one and
means workers sharing the selected GPU. Caption JSONs are under
`results/captions/audio_to_text_clean/inference_rank*/results.json`. The helper
requires one nonempty string per sample, rejects duplicate/missing captions,
and updates the manifests to use `captions.txt` instead of transcripts.

Generate audio conditioned on those rich captions:

```bash
CUDA_VISIBLE_DEVICES=0 python -m espnet2.speechlm.bin.inference \
    --train-config "$BAGPIPER_RUN/configs/inference_model.yaml" \
    --inference-config "$BAGPIPER_RUN/configs/audio.yaml" \
    --model-checkpoint "$BAGPIPER_RUN/models/bagpiper/base.pt" \
    --output-dir "$BAGPIPER_RUN/results/roundtrip" \
    --test-unregistered-specifier "text_to_audio:clean:$BAGPIPER_RUN/data/smoke/all.json"
```

This processes all 96 examples. For a first check, copy the manifest and retain
only its first `samples` entry. For parallel generation, use the same rank loop
as captioning with the audio inference configuration and `text_to_audio` task.
WAVs and an index JSON are written under `text_to_audio_clean/inference_rank*`.
CPU construction of the 8B model took about four minutes per process on this
host; the first sample's actual caption/audio decoding took about 11/9 seconds.

## 5. Eight-GPU FSDP fine-tuning and recovery

The fine-tuning inputs are **original audio + model-generated rich captions**.
Both directions are trained. The configuration trains the full decoder,
embeddings and adaptor (8.268 billion parameters); the encoder and codec remain
frozen. It uses FSDP2, eight shards, BF16 compute, FP32 parameter/Adam storage,
full activation checkpointing, and two micro-batches per optimizer step.

```bash
export BAGPIPER_TRAIN="audio_to_text:train:$BAGPIPER_RUN/data/smoke/train.json text_to_audio:train:$BAGPIPER_RUN/data/smoke/train.json"
export BAGPIPER_VALID="audio_to_text:valid:$BAGPIPER_RUN/data/smoke/valid.json text_to_audio:valid:$BAGPIPER_RUN/data/smoke/valid.json"
python -m espnet2.speechlm.bin.prepare_length_stats \
    --train-config "$BAGPIPER_RUN/configs/smoke-train.yaml" \
    --output-dir "$BAGPIPER_RUN/data/stats" --num-workers 4 \
    --train-unregistered-specifier "$BAGPIPER_TRAIN" \
    --valid-unregistered-specifier "$BAGPIPER_VALID"
```

Make all eight GPUs available; stop your serving/inference processes using them
and wait for other users' jobs to finish on shared machines.
Run from the recipe (the launcher resolves relative paths there):

```bash
cd "$BAGPIPER_REPO/egs2/bagpiper/speechlm1"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run.sh --ngpu 8 \
    --train-config "$BAGPIPER_RUN/configs/smoke-train.yaml" \
    --output-dir "$BAGPIPER_RUN/results/fsdp-smoke" \
    --resume-path "$BAGPIPER_RUN/models/bagpiper/base.pt" \
    --stats-dir "$BAGPIPER_RUN/data/stats" \
    --train-unregistered-specifier "$BAGPIPER_TRAIN" \
    --valid-unregistered-specifier "$BAGPIPER_VALID" --wandb-mode disabled
```

An explicit `--resume-path` initializes **weights only** and starts a fresh
optimizer/scheduler/step. Expected outputs are complete `step_2` and `step_3`
DCP directories, each with `.metadata` and eight `.distcp` files. Check the log
for `Loaded native model weights`, `dp_shard: 8`, finite CE/gradient norms, and
both validation tasks. The final interval must stop at step 3, not step 4.

Recover the saved run and continue to step 5 by omitting `--resume-path`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run.sh --ngpu 8 \
    --train-config "$BAGPIPER_RUN/configs/smoke-resume.yaml" \
    --output-dir "$BAGPIPER_RUN/results/fsdp-smoke" \
    --stats-dir "$BAGPIPER_RUN/data/stats" \
    --train-unregistered-specifier "$BAGPIPER_TRAIN" \
    --valid-unregistered-specifier "$BAGPIPER_VALID" --wandb-mode disabled
```

Keep the data, topology, token budget and accumulation unchanged. Recovery loads
the latest complete output checkpoint, restores Adam and the scheduler, and
continues at micro-batch 6 when optimizer step is 3 and accumulation is 2.

A final regression also restored step 5 and trained to step 6 on all eight GPUs
with alphabetically sorted YAML keys. Before the vocabulary fix, this command
silently produced CE 55.73; the corrected run produced CE 2.103 and gradient
norm 0.6141, with validation CE 0.5756 / 3.170 for audio-to-text / text-to-audio.
Its saved model, Adam and scheduler state passed the same CPU audit below.

Inspect the actual saved tensors on CPU. This helper requires finite nonzero
FP32 Adam moments, matching optimizer/scheduler/global steps, updated sampled
decoder/adaptor/stream weights, and an unchanged sampled frozen encoder weight:

```bash
cd "$BAGPIPER_REPO"
python test_utils/speechlm/inspect_bagpiper_checkpoint.py \
    --checkpoint "$BAGPIPER_RUN/results/fsdp-smoke/checkpoints/step_5" \
    --reference "$BAGPIPER_RUN/models/bagpiper/base.pt" --expected-step 5 \
    --expect-audio-input-update --output "$BAGPIPER_RUN/results/base-dcp-audit.json"
```

Export weights for native inference; this is a CPU operation:

```bash
cd "$BAGPIPER_REPO"
python -m espnet2.speechlm.bin.export_checkpoint \
    --checkpoint-dir "$BAGPIPER_RUN/results/fsdp-smoke/checkpoints/step_5" \
    --output "$BAGPIPER_RUN/models/bagpiper-smoke-step5.pt" --dtype bfloat16
```

The exporter reads only model tensors and refuses to overwrite an existing
file. BF16 casting is intentional for inference; omit `--dtype` to preserve
checkpoint dtypes. Check the training-to-inference path on the same four
validation examples used in the experiment:

```bash
python - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["BAGPIPER_RUN"])
manifest = json.loads((run / "data/smoke/valid.json").read_text())
manifest["samples"] = manifest["samples"][:4]
(run / "data/smoke/valid-four.json").write_text(json.dumps(manifest, indent=2))
PY
for pair in audio_to_text:caption text_to_audio:audio; do
    task="${pair%:*}"
    infer="${pair#*:}"
    CUDA_VISIBLE_DEVICES=0 python -m espnet2.speechlm.bin.inference \
        --train-config "$BAGPIPER_RUN/configs/inference_model.yaml" \
        --inference-config "$BAGPIPER_RUN/configs/$infer.yaml" \
        --model-checkpoint "$BAGPIPER_RUN/models/bagpiper-smoke-step5.pt" \
        --output-dir "$BAGPIPER_RUN/results/finetuned" \
        --test-unregistered-specifier "$task:valid:$BAGPIPER_RUN/data/smoke/valid-four.json"
done
```

## 6. Bagpiper-TTS dialogue recipe

After caption pairing, the helper also writes `tts_train.json`, `tts_valid.json`
and `dialogues.jsonl`: a system instruction, a user request quoting the source
transcript, an assistant text plan/caption, and the original target audio.
The short plan is synthetic and exists only to exercise the dialogue interface.

```bash
cd "$BAGPIPER_REPO"
python - <<'PY'
import os
from pathlib import Path
import yaml
run = Path(os.environ["BAGPIPER_RUN"])
config = yaml.safe_load(Path("egs2/bagpiper_tts/speechlm1/conf/train.yaml").read_text())
config["data_loading"].update(batch_size=2048, num_workers=2)
config["trainer"].update(max_step=2, save_interval=2, gradient_accumulation_steps=1)
config["trainer"]["optimizer"]["lr"] = 1e-5
config["trainer"]["lr_scheduler"].update(warmup_steps=1, decay_end_step=2)
config["trainer"]["titan_config"]["dp_shard"] = 8
(run / "configs/tts-smoke.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY
python -m espnet2.speechlm.bin.prepare_length_stats \
    --train-config "$BAGPIPER_RUN/configs/tts-smoke.yaml" \
    --output-dir "$BAGPIPER_RUN/data/tts-stats" --num-workers 4 \
    --train-unregistered-specifier "dialogue:tts_train:$BAGPIPER_RUN/data/smoke/tts_train.json" \
    --valid-unregistered-specifier "dialogue:tts_valid:$BAGPIPER_RUN/data/smoke/tts_valid.json"
cd "$BAGPIPER_REPO/egs2/bagpiper_tts/speechlm1"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run.sh --ngpu 8 \
    --train-config "$BAGPIPER_RUN/configs/tts-smoke.yaml" \
    --output-dir "$BAGPIPER_RUN/results/tts-fsdp-smoke" \
    --resume-path "$BAGPIPER_RUN/models/bagpiper-tts-sft/model.pt" \
    --stats-dir "$BAGPIPER_RUN/data/tts-stats" \
    --train-unregistered-specifier "dialogue:tts_train:$BAGPIPER_RUN/data/smoke/tts_train.json" \
    --valid-unregistered-specifier "dialogue:tts_valid:$BAGPIPER_RUN/data/smoke/tts_valid.json" \
    --wandb-mode disabled
```

Export this fine-tune and check a native text/audio dialogue before serving:

```bash
cd "$BAGPIPER_REPO"
python -m espnet2.speechlm.bin.export_checkpoint \
    --checkpoint-dir "$BAGPIPER_RUN/results/tts-fsdp-smoke/checkpoints/step_2" \
    --output "$BAGPIPER_RUN/models/bagpiper-tts-smoke-step2.pt" --dtype bfloat16
python test_utils/speechlm/inspect_bagpiper_checkpoint.py \
    --checkpoint "$BAGPIPER_RUN/results/tts-fsdp-smoke/checkpoints/step_2" \
    --reference "$BAGPIPER_RUN/models/bagpiper-tts-sft/model.pt" --expected-step 2 \
    --output "$BAGPIPER_RUN/results/tts-dcp-audit.json"
python - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["BAGPIPER_RUN"])
manifest = json.loads((run / "data/smoke/tts_valid.json").read_text())
manifest["samples"] = manifest["samples"][:1]
(run / "data/smoke/tts-valid-one.json").write_text(json.dumps(manifest, indent=2))
PY
CUDA_VISIBLE_DEVICES=0 python -m espnet2.speechlm.bin.inference \
    --train-config "$BAGPIPER_RUN/configs/inference_model.yaml" \
    --inference-config "$BAGPIPER_RUN/models/bagpiper-tts-sft/inference.yaml" \
    --model-checkpoint "$BAGPIPER_RUN/models/bagpiper-tts-smoke-step2.pt" \
    --output-dir "$BAGPIPER_RUN/results/tts-finetuned-native" \
    --test-unregistered-specifier "dialogue:valid:$BAGPIPER_RUN/data/smoke/tts-valid-one.json"
```

The TTS-only smoke inputs contain no user audio. Its unused continuous-audio
adaptor need not change; the decoder and audio stream weights do update.

## 7. Independent vLLM installation and serving

Keep the training environment intact. In another shell, set `BAGPIPER_RUN`,
`BAGPIPER_REPO`, `BAGPIPER_PYTHON` and `HF_HOME` to the values above. Do not leave
`HF_HUB_OFFLINE=1` set during first
conversion/serving, because the converter fetches public assets and audio output
may download the codec. Use the [ESPnet fork](https://github.com/espnet/vllm),
whose [Chinese guide](https://github.com/espnet/vllm/blob/main/examples/espnet/GETTING_STARTED.zh.md)
documents its audio-specific runtime and clients.

```bash
"$BAGPIPER_PYTHON" -m pip install uv
unset HF_HUB_OFFLINE
git clone https://github.com/espnet/vllm.git "$BAGPIPER_RUN/source/vllm"
cd "$BAGPIPER_RUN/source/vllm"
git checkout 818d406909c1241f151966d97ea6e88f5edea0d8
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip check
uv pip freeze > "$BAGPIPER_RUN/logs/vllm-environment.txt"
.venv/bin/python examples/espnet/convert/convert_bagpiper_ckpt.py \
    "$BAGPIPER_RUN/models/bagpiper-tts-sft/model.pt" \
    "$BAGPIPER_RUN/models/bagpiper-tts-vllm"
```

The converter writes config/tokenizer files and four safetensors shards from
the native checkpoint. It validates known tensor prefixes and deliberately
drops only the training loss buffer `vocab_weight`. The measured conversion
retained all 1381 model tensors bit-for-bit. This fork declares a v0.28.0 base;
a shallow editable clone may report a different development version string,
so record the source commit as well as resolved package versions.

Start the service after eight-GPU training finishes:

```bash
CUDA_VISIBLE_DEVICES=7 MODEL_PATH="$BAGPIPER_RUN/models/bagpiper-tts-vllm" \
    HOST=127.0.0.1 PORT=19811 LOGFILE="$BAGPIPER_RUN/logs/vllm-server.log" \
    bash examples/espnet/serve_bagpiper.sh --max-model-len 8192 --max-num-seqs 4
```

Wait for the API to be ready. The official launcher selects the fork's supported
model runner and scheduling mode. In another shell using the same venv:

```bash
cd "$BAGPIPER_RUN/source/vllm"
.venv/bin/python examples/espnet/clients/client_bagpiper.py \
    --task tts --port 19811 --max-tokens 2048 \
    --out "$BAGPIPER_RUN/results/vllm-default.wav"
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import subprocess
import sys
run = Path(os.environ["BAGPIPER_RUN"])
first = (run / "data/smoke/captions.txt").read_text().splitlines()[0]
caption = first.split(maxsplit=1)[1]
subprocess.run([
    sys.executable, "examples/espnet/clients/client_bagpiper.py",
    "--task", "tts_cfg", "--cfg", "3", "--port", "19811",
    "--max-tokens", "4096", "--prompt", caption,
    "--out", str(run / "results/vllm-generated-caption.wav"),
], check=True)
PY
```

Use the client's default audio-generation system prompt. Put quoted speech
inside a scene description (the rich caption already does this). A successful
check includes stop termination, a nonempty waveform and content inspection;
HTTP 200 alone is insufficient. Native Base generation and TTS-SFT serving use
different checkpoints and prompts, so they are not expected to be bit-identical.

To serve the TTS fine-tune exported in section 6, use the vLLM venv to convert
it, reusing the validated tokenizer/config assets:

```bash
cd "$BAGPIPER_RUN/source/vllm"
.venv/bin/python examples/espnet/convert/convert_bagpiper_ckpt.py \
    "$BAGPIPER_RUN/models/bagpiper-tts-smoke-step2.pt" \
    "$BAGPIPER_RUN/models/bagpiper-tts-smoke-vllm" \
    --ref-dir "$BAGPIPER_RUN/models/bagpiper-tts-vllm"
```

Stop the existing service, set `MODEL_PATH` to the new directory, and rerun the
same launcher/client. The experiment verified this complete chain: native
eight-GPU training → DCP → native model file → vLLM conversion → generated audio.

For development checks of this fork, follow its `AGENTS.md`/contributing setup
with `uv`, then run the focused Bagpiper test:

```bash
uv pip install -r requirements/lint.txt
.venv/bin/pre-commit install
uv pip install pytest tblib pytest-asyncio pytest-forked pytest-timeout pytest-mock
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -o addopts= -q \
    tests/model_executor/test_bagpiper.py
```

## 8. Inspect and retain the evidence

This included helper checks every WAV against the selected IDs. With `--asr`, it
downloads TorchAudio's public wav2vec2 ASR model and transcribes on CPU:

```bash
cd "$BAGPIPER_REPO"
export TORCH_HOME="$BAGPIPER_RUN/torch"
"$BAGPIPER_PYTHON" test_utils/speechlm/evaluate_bagpiper_smoke.py \
    --samples "$BAGPIPER_RUN/data/smoke/samples.json" \
    --results "$BAGPIPER_RUN/results/roundtrip" --expected-count 96 --asr \
    --output "$BAGPIPER_RUN/results/roundtrip-check.json"
```

The measured Base roundtrip produced 96/96 valid mono 16 kHz WAVs, totaling
447.92 s. Greedy ASR gave 72 word edits over 1425 reference words (5.05%): a
diagnostic for this subset, including spelling/segmentation and ASR errors.
Four held-aside smoke examples generated with the fine-tuned Base weights also
passed waveform checks (2 ASR edits / 50 words). Native fine-tuned TTS produced
both a text plan/caption and audio; its checked six-word sentence transcribed
exactly. The fine-tuned vLLM request produced 4.44 s audio with one ASR insertion
on the 17-word sentence. These smoke updates do not establish quality gains.

Retain the source commits, package freezes, checkpoint hashes, generated configs,
sample IDs, captions, WAVs, loader assignments and full command output. The
validation run's artifact root is
`/mnt/project/jinchuan/espnet_sync/.artifacts/bagpiper-e2e-20260909`:

- `commands.jsonl` and `logs/`: commands, working directories, UTC times,
  durations, exit codes, complete output, and per-GPU inference logs.
- `data/smoke/samples.json`: source audio, original transcripts and split IDs.
- `results/fsdp-checkpoint-audit.json`: actual parameter deltas, unchanged
  frozen parameter, FP32 Adam moments and matching saved step counters.
- `results/vllm-conversion-check.json`: exact native/safetensors comparison.
- `results/audio-file-check.json` and `results/audio-asr.json`: waveform and
  independent CPU ASR checks. The first LibriSpeech sentence regenerated by
  native Base and vLLM CFG transcribed exactly to its reference; this single
  sentence does not establish general audio quality or benchmark accuracy.
- `results/sorted-inference-comparison.json`: the four checked captions and
  four audio waveforms remain exactly equal when YAML mappings are reordered.
- `results/final-recovery-check.json` and `results/final-step6-dcp-audit.json`:
  successful eight-GPU recovery with the formerly failing sorted YAML, finite
  training/validation metrics, eight saved shards and verified step-6 state.
- `bagpiper-validation-evidence.tar.gz`: a compact archive of commands, logs,
  configs, sample manifests and JSON checks, with per-file SHA-256 hashes in
  `evidence-manifest.json`; it excludes checkpoints, environments and audio.

The [experiment log](EXPERIMENT_LOG.md) links these observations to the fixes.
Large models, corpus files and generated experiment artifacts stay outside Git.
