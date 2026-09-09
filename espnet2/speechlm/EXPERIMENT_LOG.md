# Bagpiper end-to-end validation log

This records a clean-environment validation begun on 2026-09-09 (UTC).
Results are added only after the corresponding command runs. The
[reproduction guide](REPRODUCE.md) distinguishes a smoke experiment from a paper
benchmark. LibriSpeech test-clean is deliberately used for the requested smoke
fine-tuning; its post-training results must not be reported as held-out scores.

## Workspace and hardware

- Upstream baseline: `4fb9218b9b6427e3c4dc1582958facbf17298e4e`.
- Branch: `bagpiper-e2e-validation`, created from fresh `origin/master`.
- Installation PR A (#6645) was already merged at that baseline. Running
  `git merge jinchuan/A-installation` reported `Already up to date.`
- Trainer PR B (#6646) and recipe PR C (#6647) are also included.
- Worktree: `/mnt/project/jinchuan/espnet_sync/.worktrees/bagpiper-e2e`.
- Artifacts: `/mnt/project/jinchuan/espnet_sync/.artifacts/bagpiper-e2e-20260909`.
  Raw command output is in `logs/`; `commands.jsonl` records command, working
  directory, UTC start/end time, duration, and exit status. Large model/data/log
  artifacts are not included in Git.
- Eight NVIDIA H100 80 GB HBM3 GPUs, initially idle (GPU 0: 528 MiB;
  remaining GPUs: 3–4 MiB), 224 logical CPUs, approximately 2 TiB host RAM.
- CUDA compiler: 12.9.41, `/mnt/project/jinchuan/tools/cuda-12.9/bin/nvcc`.
- The first sandboxed `nvidia-smi` reported that it could not communicate with
  the driver. Repeating the read-only check with host GPU access succeeded;
  this was execution isolation, not an installation defect.
- Existing environments and untracked research files were left untouched.

## 1. Clean environment and upstream assets

1. Ran `git fetch origin master` and `git fetch jinchuan A-installation`.
2. Created the isolated worktree with `git worktree add -b
   bagpiper-e2e-validation <worktree> origin/master`; merged PR A's final head.
3. Read the public collection API and model cards at
   <https://huggingface.co/collections/espnet/bagpiper>.
   - `espnet/bagpiper`, revision `cde214717b75190c0b117c0f3ad59b1412e3c695`,
     `base.pt`, SHA-256
     `c9917c1237bd44ea0ea8b41746e5ec5aecae92273538c9b6e4040d26f69a6521`.
   - `espnet/bagpiper-sft`, revision
     `5646600302bf2670214c601721e44f6d36145c26`, `model.pt`, SHA-256
     `50980756fad38bbefeb326ac0012e39f431ec5af8c0caaac8bd254f179f33746`.
     This entry was inspected in the collection metadata; its weights were
     not downloaded or tested. The actual runs use Base and TTS-SFT below.
   - `espnet/bagpiper-tts-sft`, revision
     `675e2fafccc7dd7205fad6f8fdc4451f9ee6f768`, `model.pt`, SHA-256
     `168fe0aca32f4b1636e4ee2e63ef3880427be352bd6369a1f5ba3b49750f91da`.
4. Located the official implementation at <https://github.com/espnet/vllm>.
   Its main head was `818d406909c1241f151966d97ea6e88f5edea0d8` and its
   declared upstream wheel version was vLLM 0.28.0. Its README requires a
   conversion from native ESPnet weights, with public tokenizer/config assets.
5. Created a fresh environment:
   `tools/miniconda3/bin/conda create -n bagpiper-e2e python=3.12 pip -y`.
   Result: success; raw log `001-conda-create.log`.

Initial code inspection identified two cases to reproduce before fixing:
the released `.pt` files are unsupported by Titan's DCP-only initialization,
and the fused loss unpacks three Liger outputs despite newer Liger versions
returning four. Neither has been treated as validated by inspection alone.

## 2. Installation and baseline failures

The numbered names below refer to raw logs in the artifact directory. Each
record in `commands.jsonl` includes the exact command, environment overrides,
working directory, start/end times, and exit code.

| Logs | Action and observed result | Resolution |
| --- | --- | --- |
| 003, 006 | The documented PyTorch install inherited an unreachable `pypi.ngc.nvidia.com` extra index and retried every package. Stopped this installer after 218 s. | Removed that extra-index environment variable and set `PIP_CONFIG_FILE=/dev/null` for the command. Official cu128 install succeeded in 209 s. No global pip configuration was edited. |
| 010 | `pip install -e '.[speechlm]'` on the new worktree succeeded in 211 s. | Resolved PyTorch 2.11.0+cu128, Transformers 5.14.1, TorchTitan 0.2.2, Liger 0.8.2. |
| 017, 018, 022 | Installed build/test tools, built current FA3 from source, and installed causal-conv1d. | FA3 commit `9d61d35ba876834539af7df369fbabef1ad1e5d7`, complete build with `MAX_JOBS=32 NVCC_THREADS=2`, 1406 s, wheel SHA-256 `b10242f707612aa36082a15990bb293c9758fa98574060e96b32091a221fa406`. causal-conv1d 1.7.0 installed in 3.4 s. |
| 021, 043 | Dependency check and actual FA3 BF16 forward/backward. | Both passed; Transformers recognizes FA3. |
| 007, 014, 020 | Downloaded OpenSLR test-clean in 13.5 s. Initial tar extraction failed restoring archive ownership on the managed filesystem. | `tar --no-same-owner` succeeded. The files are the official archive, not an existing local corpus. |
| 008, 009, 011, 027, 031 | Downloaded Base and TTS checkpoints plus model configurations. One initial urllib YAML request encountered a connection reset. | Retried public raw URLs with curl retry/timeout. Both full checkpoint SHA-256 values match the public cards. |
| 019, 025 | All SpeechLM tests initially failed collection: a dataloader conftest installed a fake PyArrow over the real package, breaking pandas/Transformers imports. | Changed fallback detection to import the real dependency first; 369 tests passed in 19.65 s. |
| 023 | Single-GPU inference with default rank crashed: `NoneType -= int`. | Added usable rank/world/worker defaults, validation, and nonzero failure propagation from subprocesses. Removed unconditional 60 s startup delay and nested output paths derived from absolute manifest paths. |
| 024, 026 | Four real CUDA loss comparisons failed with `too many values to unpack (expected 3)`. | Consume the first three documented Liger results; ignore its optional prediction output. Also guard zero z-loss weight and include multimodal z-loss in statistics. All four forward/gradient comparisons against PyTorch passed. |
| 029 | Passing the real `base.pt` to `_load_checkpoint` printed `No checkpoint found, starting from step 0`, without touching any model. | Added strict native state-dict initialization using PyTorch's distributed state-dict API, loading once on rank 0 and distributing to FSDP shards. Missing explicit paths now raise errors. Actual distributed validation is recorded in logs 061 and 072. |
| 033, 039, 041 | Fetched only upstream configuration/tokenizer files. Tested the standalone Qwen3 audio tower and worker copies. First manual call used FP32 features against BF16 weights and failed the dtype contract. | Retested with BF16 features as used by inference/training: returned 52 × 2048 features for 400 mel frames. Worker copy has no model, original retains its model. |
| 034, 038, 044 | Source changes passed 369 tests; a new empty-checkpoint test exposed an unhelpful PyTorch error without a process group. | Reject empty native state dicts explicitly; all 18 focused loader/CLI/trainer regressions passed. |
| 035 | Prepared fixed test-clean subset by round-robin speaker selection, sorted IDs, 2–10 s clips. | 96 clips, 40 speakers, 543.3550625 s; first 64 for smoke training and remaining 32 for validation. Transcripts are kept for inspection; generated captions will condition fine-tuning. |

## 3. Independent vLLM environment

- Created `source/vllm/.venv` with `uv venv --python <new Python 3.12>`.
  Installed with `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`
  (log 005): success in 319 s. This environment is separate from training.
- Resolved PyTorch 2.13.0+cu132, Transformers 5.16.1. The shallow clone reports
  version `0.1.dev1+g818d40690`, despite the fork declaring a v0.28.0 base.
  Actual source commit and resolved packages are recorded; serving verification
  is required, not a version-string assumption.
- Log 012 tried the outdated `vllm._C` module and failed. Log 016 used the
  current `_custom_ops` entrypoint and imported the Bagpiper model successfully;
  8 GPUs visible. This was an incorrect diagnostic, not a missing extension.
- Installed repository-required lint/test tools and hooks (013).
- Dependency check first hit the sandbox's read-only default uv cache (032).
  Repeated with an explicitly writable cache (037): passed.
- Converted the downloaded `espnet/bagpiper-tts-sft` using the fork's converter
  (030): 1381 BF16 tensors in 4 safetensors shards; only the documented
  training-only `vocab_weight` buffer is dropped. Tokenizer/config assets came
  from the converter's pinned public sources.
- Started the real server on GPU 7, loopback-only `127.0.0.1:19811`, using the
  official launcher with 8192-token context and maximum 4 sequences (036).
  Weight loading used 16.63 GiB; compile and CUDA graph warmup completed.
- First official client TTS request (040): HTTP 200, `finish_reason=stop`,
  918 output tokens including the text plan, 1.40 s mono 16 kHz WAV. Prompt is
  the client's default scene containing “Hello, how are you today?”.
  Output is `results/vllm-default.wav`; no audio-quality claim is inferred from
  successful HTTP or file creation alone.

## 4. Released-model inference and prepared smoke data

| Logs | Action and observed result | Resolution / artifact |
| --- | --- | --- |
| 042 | Strictly loaded all 1382 tensors from the public Base checkpoint and captioned `61-70968-0000`. | The caption includes the full reference sentence and a description of the speaker/acoustics. Model creation takes about four minutes on CPU, then decoding about 11 seconds. Config-only construction avoids downloading a redundant Qwen3-8B plus Qwen3-Omni checkpoint. |
| 045, 046, 048 | The fork's Bagpiper tests initially required the test-only package `tblib`. | Installed its focused test dependencies in the separate vLLM environment; `tests/model_executor/test_bagpiper.py`: 2 passed. |
| 047, 052 | Launched six independent native inference shards on GPUs 0–5 (one worker per GPU), then paired captions with the source audio. | 96/96 nonempty captions, no duplicate IDs; 454.7 s including model initialization. Native text JSON now stores a string, matching text dataset input, instead of a one-element list. `data/smoke/{all,train,valid}.json` now point to generated `captions.txt`. |
| 050 | Official vLLM CFG client, scale 3, with a scene quoting the first LibriSpeech sentence. | HTTP success, stop termination, 4.00 s mono 16 kHz audio. |
| 051 | Real CPU DCP export regression, including an optimizer entry. | Passed: exported native file contains only exact model weights and refuses to overwrite an output file. |
| 053 | Made short configs from the Bagpiper SFT recipe. | 2048 packed token slots per GPU, accumulation 2, 8-way FSDP, BF16 compute, FP32 storage, all decoder parameters trainable, frozen audio encoder/codec. Initial `max_step=3`, `save_interval=2`; resume target 5. |
| 054 | Native Base caption-to-audio on the first generated rich caption. | Success in 270.2 s including CPU model initialization; decoding about 9 s. Output: 4.96 s, 16 kHz mono, finite/nonzero samples. |
| 055 | Ran the public length-statistics CLI for both directions and both splits, with four CPU workers. | All four stats files written in 37.0 s; 64 entries per training task and 32 per validation task. No full upstream model weights fetched. |
| 056 | Sent the actual model-generated rich caption to the official vLLM CFG client. | Stop termination, 1117 completion tokens, 4.38 s audio; request completed in 10.6 s. |
| 057, 058 | Behavioral regression reproduced two trainer errors: target step 3 ended at 4, and resume at step 3 with accumulation 2 read micro-batches 3–6 instead of 6–9. | Bound the final save interval by remaining steps and convert optimizer-step offsets to micro-batch offsets. All 30 focused trainer/CLI/checkpoint tests passed. Automatic resume also ignores incomplete checkpoint directories and constructs from config when a complete output DCP will restore all weights. |
| 059, 060 | Stopped only this experiment's vLLM API process to free GPU 7; checked all four WAV files. | Finite, nonempty, 16 kHz mono waveforms. File statistics are in `results/audio-file-check.json`. This check does not establish semantic fidelity. |

## 5. Eight-GPU training

Log 061 launches the actual recipe `egs2/bagpiper/speechlm1/run.sh --ngpu 8`
with the downloaded Base `.pt`, prepared paired manifests, and new environment.
All eight ranks initialized NCCL. Training/restore results are recorded below;
successful initialization alone is not a training pass.

- Log 061 completed successfully in 369.0 s: 8.268 billion trainable parameters,
  `dp_shard=8`, `dp_replicate=1`, accumulation 2, BF16 compute and FP32 parameter
  storage. Three optimizer steps completed, with finite cross-entropy
  1.784 / 1.671 / 1.806 and gradient norms 0.9274 / 0.9823 / 0.9165. The first
  step took 5.86 s; subsequent steps took 1.60 / 1.55 s. Both directions were
  validated, and DCP directories `step_2` and `step_3` each contain eight shards.
- Log 064 reran the same recipe with `max_step=5`, the same output directory,
  and **no `--resume-path`**. Completed in 403.6 s, with no network access to
  fetch backbone weights. It restored step 3, scheduler epoch 3 and LR
  `6.890576474687264e-6`, used micro-batch offsets 6–9, and saved `step_5`.
  Subsequent CE values were 1.267 / 1.789, with finite gradient norms
  1.213 / 0.8685. These few steps validate execution, not convergence.
- Log 065 independently inspected the real DCP metadata and global step.
  An empty dictionary is not a schema for automatically reading scheduler
  entries: the explicit schema in log 067 reads and verifies those entries.
- Log 066 exported all 1382 model tensors from the eight-shard `step_5` DCP to
  `models/bagpiper-smoke-step5.pt` in 60.6 s, casting floating tensors to BF16
  for inference. Optimizer tensors were not exported.
- Log 067 read actual model/Adam/scheduler tensors from step 3 and step 5.
  Adam moment tensors are finite FP32 with nonzero entries; optimizer step,
  global step and scheduler epoch agree (3 then 5). The first decoder Q
  projection, audio adaptor and stream embeddings have changed from the
  downloaded Base weights (max absolute change about `3.0e-5` at step 5).
  The sampled frozen audio encoder projection is exactly unchanged.
  Full audit results: `results/fsdp-checkpoint-audit.json`.
- Log 068: all five real CUDA tests passed, including Liger loss/gradient
  comparisons and actual Transformers grouped-MM dispatch against eager MoE.

## 6. Independent inference checks

- Log 062 downloaded TorchAudio's public `WAV2VEC2_ASR_BASE_960H` and ran greedy
  CTC recognition on CPU. On the first LibriSpeech sentence, the original
  recording, native Base regeneration, and both vLLM CFG regenerations all
  transcribed to the 17-word reference exactly. This is a one-sentence content
  check, not a WER benchmark or a judgment of voice/prosody quality.
  The short default greeting transcribed as “HULLO HOW ARE YOU TO DAY”, giving
  3 edit errors against “HELLO HOW ARE YOU TODAY”; spelling/segmentation affects
  this small diagnostic. Raw transcriptions: `results/audio-asr.json`.
- Log 063 compared every converted safetensors tensor against the native
  downloaded TTS checkpoint: all 1381 retained tensors are bit-exact, with
  identical dtypes. Only the documented loss buffer `vocab_weight` is absent.
  Result: `results/vllm-conversion-check.json`.
- Logs 069–071 add smoke-only instruction dialogues using the same audio and
  generated captions, then collect dialogue lengths for the TTS recipe.
  The plan prefix is synthetic and these examples do not reproduce the paper's
  instruction dataset. The dialogue path also exercises consecutive assistant
  text/audio turns and a data loader with two worker processes per rank.

## 7. TTS fine-tuning, export, and serving the fine-tune

| Logs | Action and measured result | Evidence / limitation |
| --- | --- | --- |
| 072 | Ran `egs2/bagpiper_tts/speechlm1/run.sh --ngpu 8` with the downloaded TTS-SFT `.pt`, 2048-token packing, two workers per rank, two optimizer steps. Completed in 443.5 s. | CE 1.917 / 2.360, gradient norm 1.643 / 1.455, dialogue validation CE 1.814. Saved eight DCP shards at `results/tts-fsdp-smoke/checkpoints/step_2`. Full decoder fine-tuning, with encoder/codec frozen. |
| 073 | Checked every tensor of the Base step-5 export against the native checkpoint schema. | All 1382 names/shapes match and all values are finite. This checks completeness, not equality to the pre-training weights. |
| 074 | Generated the complete Base caption-to-audio roundtrip on four GPUs, plus four validation examples in each direction with the exported Base fine-tune. All six subprocesses completed in 437.9 s. | 96/96 roundtrip WAVs; 4/4 fine-tuned captions and 4/4 fine-tuned WAVs. Per-process commands/output are in `logs/roundtrip-gpu*.log`, `finetuned-caption.log`, and `finetuned-audio.log`. |
| 075, 077 | Exported the TTS step-2 DCP to native BF16 weights (53.2 s), then ran the fork's converter (46.4 s). | `models/bagpiper-tts-smoke-step2.pt` and `models/bagpiper-tts-smoke-vllm`. Conversion reuses the original conversion's validated tokenizer/config files via `--ref-dir`. |
| 076 | Ran all SpeechLM unit tests after the main fixes. | 379 passed in 20.70 s; later validation fixes add more tests below. |
| 078, 079 | Saved both complete package freezes. | Training uses torch 2.11.0+cu128 / Transformers 5.14.1; vLLM uses torch 2.13.0+cu132 / Transformers 5.16.1. Separate environments prevent one install from replacing the other's binary dependencies. |
| 080 | Native inference of the exported TTS fine-tune on one validation dialogue, including text plan/caption and audio. | Completed in 282.0 s, including CPU model construction. Output under `results/tts-finetuned-native/dialogue_valid/inference_rank0`. |
| 081, 084 | Started a new vLLM server on GPU 7 / loopback port 19812 using the **fine-tuned** TTS export, then requested audio from the actual generated LibriSpeech caption with CFG 3. | Request completed in 16.2 s: stop termination, 965 completion tokens, 4.44 s mono 16 kHz WAV at `results/vllm-finetuned-caption.wav`. This completes training → DCP → native export → conversion → real serving. |
| 086 | Stopped this experiment's fine-tuned vLLM API process after the request. | Released the GPU for the subsequent eight-GPU check; no other service was interrupted. |

## 8. Waveform and content diagnostics

- Logs 085 and 088 checked all 96 roundtrip files: each is nonempty, finite,
  non-silent, mono 16 kHz. Total duration 447.92 s, shortest 0.62 s and longest
  10.0 s. The tracked `test_utils/speechlm/evaluate_bagpiper_smoke.py` helper
  independently reproduces the checks in 12.7 s on this host.
- TorchAudio's public wav2vec2 recognizer with greedy CTC produced 72 word edits
  over 1425 reference words (5.0526%) for these 96 Base regenerations.
  This is a content diagnostic; it includes recognizer, spelling and
  segmentation errors and does not measure caption grounding, speaker
  similarity or prosody. No voice-quality or paper-benchmark claim follows.
- Four fine-tuned Base regenerations passed waveform checks: 15.58 s total,
  2 word edits / 50 reference words. One native fine-tuned TTS output matched
  its six-word reference in this recognizer.
- Log 089 checked the fine-tuned vLLM output: one insertion over the 17-word
  reference (the recognizer produced “COMPLAINT PLAINT”). This is **not** exact
  transcription and does not establish an improvement from two training steps.
- Detailed per-sample output is retained in `results/roundtrip-audit.json`,
  `results/roundtrip-repro-check.json`, and `results/finetuned-vllm-asr.json`.

## 9. Failures found during the final training review

### Validation conditioning dropout

- Log 082 reproduced validation using the training preprocessor's random
  classifier-free guidance dropout. Setting CFG dropout to 1 in the regression
  removed the user conditioning, so validation was partly unconditional.
- Source fix: build a separate validation preprocessor with `audio_cfg=0`,
  preserving assistant targets and the requested packing. Log 083: all 60
  focused job-template/training CLI tests passed. Earlier validation CE values
  above are historical measurements with the old dropout behavior.

### YAML mapping order silently changed the token vocabulary

- Log 087 attempted another eight-GPU recovery from good step 5. The generated
  configuration used default `yaml.safe_dump` sorting. The job exited **0**,
  but its optimizer-step CE jumped to **55.73**, gradient norm to **86.36**,
  audio-to-text validation CE to **10.39**, and text-to-audio CE to **87.30**.
  This run is a semantic failure, not a pass; its filename reflects the
  validation experiment's original intent.
- Root cause: `_build_vocabulary` iterated YAML insertion order. Alphabetical
  keys put discrete audio before text. Instead of text `[256, 152192)` and
  audio `[152192, 160392)`, it assigned audio `[256, 8456)` and text
  `[8456, 160392)`. The total vocabulary size and checkpoint shapes were
  unchanged, so strict tensor loading could not detect the wrong token IDs.
- Log 090 reproduced the error without a GPU: three of six possible IO key
  orders changed the vocabulary. The regression compares the complete token
  layout and preprocessed input/target IDs in **both** task directions.
- Source fix: vocabulary construction explicitly puts special tokens first,
  then text, then other discrete modalities ordered by name, matching every
  released recipe/checkpoint. Mapping order no longer defines token IDs.
  Vocabulary intervals are also written to the normal initialization log.
  Log 091: all 66 focused tests passed, including all six orders.
- Log 092 preserved the anomalous eight-shard `step_6` under
  `results/quarantine/yaml-vocabulary-order/step_6`, outside automatic resume
  discovery. Its metadata hash is unchanged and `reason.json` identifies
  the failed run. Good step 5 and all prior model exports remain intact.
- Log 093 reran **the same sorted-YAML command** from log 087 on the corrected
  source and good step 5. It initially waited for another job on GPUs 1–4.
  After all eight GPUs became idle it launched, but a different process
  occupied those GPUs during CPU model construction. FSDP initialization
  failed with CUDA OOM (other-process usage 75.9–77.1 GiB). The corrected text
  and audio intervals were printed before the failure; no optimizer step ran.
  This is resource contention, not a validated training result. No other
  user's process was interrupted.
- Log 095 waited for a stable two-minute idle window. The other training job
  finished, but the platform's periodic `/usr/local/bin/cudaCheck` context kept
  resetting the overly strict "no process" check. Stopped only our waiting
  launcher after 54.6 minutes; it had not started any training subprocess.
  Log 109 filters this identified health check while still rejecting other GPU
  processes, then retries the unchanged sorted-YAML command from good step 5.
- Log 109 completed the corrected eight-GPU run successfully in 638.2 s,
  including the idle wait. It restored global step 5, scheduler epoch 5 and
  LR `1.8594235253127373e-6`, consumed micro-batches 10 and 11, and saved a
  complete eight-shard `step_6`. With the same sorted configuration and good
  step-5 checkpoint as the failed run, training CE is **2.103** and gradient
  norm **0.6141**. Audio-to-text validation CE is **0.5756** and text-to-audio
  CE is **3.170**, with validation CFG dropout disabled.
- Log 110 checked the actual training output and confirmed the configuration
  SHA-256 still equals
  `21b773558b7c06fed7ca34dbac1e8f00683a59f313ed6ba937b16b68c1231e73`.
  This verifies the vocabulary repair using the same input that previously
  produced CE 55.73. Result: `results/final-recovery-check.json`.
- Log 111 independently read the saved good `step_6` DCP: all eight shards
  exist, global/scheduler/Adam step counters equal 6, and sampled Adam moments
  are finite, nonzero FP32. Sampled decoder, audio adaptor and stream weights
  changed by up to `3.2e-5` from the public Base weights; the sampled frozen
  encoder weight is exactly unchanged. Result:
  `results/final-step6-dcp-audit.json`. The anomalous step 6 remains quarantined.

## 10. Final code and guide checks

- Log 094 checked all 21 shell blocks in the first complete guide and parsed
  every embedded Python snippet. It actually executed the two configuration
  writers in a separate artifact directory: the generated training and
  sampling configurations equal those used in the successful experiments.
  Native inference architecture/preprocessing settings also match.
- Log 096: 46 audio-IO/export regressions passed, including two worker-copy
  checks that preserve the original model, retain preprocessing behavior, and
  carry no parameters into workers. CPU DCP export checks BF16 conversion
  while preserving integer buffer dtype and excluding optimizer state.
- Logs 097 and 098: Black's multiprocessing formatter stalled inside the
  execution sandbox, first with the 224-worker host default, then with one
  worker. Stopped only those formatter processes. Log 099 uses sequential
  per-file Black, target Python 3.12 and a writable cache; formatting and checks
  passed in 12.4 s. The repository's pinned Black 26.5.1 / isort 9.0.1 were
  used. No packaging configuration was changed.
- Log 100: **389 SpeechLM tests passed** in 16.91 s after all source fixes and
  formatting. Five real CUDA kernel tests passed in log 068; the two focused
  vLLM Bagpiper tests passed in log 048. `git diff --check` and shell syntax
  checks of the shared launcher and both recipe entrypoints also passed.
- Log 101 rechecked the expanded guide: 22 shell blocks, four executed
  config/manifest writers, and exact agreement of the four-example Base and
  one-example TTS inference manifests with the actual experiment.
- Logs 102–104 turn the DCP audit into the tracked
  `test_utils/speechlm/inspect_bagpiper_checkpoint.py` helper. Base step 5
  passes in 7.3 s. The first TTS audit incorrectly required the continuous-audio
  adaptor to change, although TTS smoke inputs have no user audio. Corrected
  the audit to make that requirement explicit for audio-input training only.
  TTS step 2 then passes in 4.3 s: decoder/stream weights changed by up to
  `1.0014e-5`, sampled encoder and unused adaptor remained unchanged, Adam
  moments are finite nonzero FP32, and all saved step counters equal 2.
- Log 105 uses the free GPUs 5 and 6 while other jobs still occupy 1–4. It runs
  both native directions on the four previously checked validation examples,
  with the good step-5 export and a sorted YAML. It compares caption strings
  and decoded waveform samples to the original-order run: **all four captions
  and all four waveforms are exactly equal**. Both subprocesses exited 0 after
  571.2 s, including model construction and process/driver cleanup. Results:
  `results/sorted-inference-comparison.json`.
- Logs 106–107 stage only code/regression files and create the local source-fix
  commit `6239adaf90` on `bagpiper-e2e-validation`. Documentation and smoke audit
  helpers are prepared separately; no model, corpus or generated audio is added
  to Git.
- Log 108 rechecks the final guide including CPU checkpoint audits: 23 shell
  blocks pass syntax checks and all four config/manifest writers still match
  the actual experiment. The three reproduction helpers pass formatting.

All repairs above are source changes. Real installation, inference, training,
export and serving commands ran without runtime model patches. Unit-test mocks
are limited to the inexpensive regression fixtures.

## Validation scope

The experiments cover the public Base and TTS-SFT Qwen3-8B/Xcodec checkpoints,
native captioning/audio generation, both prepared-data recipes with eight-GPU
FSDP, automatic training recovery, CPU weight export, and single-GPU serving
through the ESPnet vLLM fork. They use local disabled W&B logging; no experiment
was uploaded. Pipeline parallelism, multiple nodes, other GPUs/backbones,
long-run convergence and paper benchmark quality were not measured here.
Native weight-file initialization explicitly requires `pp_degree=1`.

The unmodified upstream master and A/B/C branches are preserved. The new
branch contains the tested source fixes and this reproduction documentation;
`pyproject.toml` is unchanged. Full environments, checkpoints, generated audio,
and the failure evidence remain under the artifact directory for inspection.
All training, inference and serving processes launched for this validation have
exited; the experiment no longer occupies GPUs.
