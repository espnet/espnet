# WavLM-large multilingual pretraining — working notes

This fork of espnet runs WavLM-large SSL pretraining on a 34-corpus multilingual
set: **62,483,742 utterances / 149,616.6 h**, against LibriSpeech 960h's 281k
utterances. Almost everything unusual in this repo exists because of that ~220x
scale difference. Read this before changing anything in the training path.

Recipe lives in `egs2/librispeech/wavlm1/` (the `librispeech` directory name is
vestigial — the data is not LibriSpeech).

---

## Where things live

Data and experiments are **outside the repo**, in a shared writable tree so
teammates can access them:

| what | path | size |
|---|---|---|
| dump (wav.scp, shapes, km labels) | `/mnt/weka/data/wavlm_expts/dump` | 841 GB |
| experiments / checkpoints | `/mnt/weka/data/wavlm_expts/exp` | 376 GB |
| token lists | `egs2/librispeech/wavlm1/data/` | small, in repo tree |
| source audio | `/mnt/weka/data/tagger_data/pretraining/audio` | — |

Python: `/mnt/weka/mark/espnet/tools/anaconda/envs/espnet/bin/python`
(torch 2.9.1+cu128). There is no `python` on PATH; use the full path.

---

## Running it

Iteration 0 of the 3-iteration pipeline, resuming from the latest checkpoint:

```bash
cd egs2/librispeech/wavlm1
./run_large.sh --stage 7 --stop_stage 7 --train_start_iter 0 --train_stop_iter 0
```

`--resume true` is passed automatically, so this picks up from
`checkpoint.pth` in the exp dir. The exp dir also holds a generated `run.sh`
with the exact full invocation — prefer copying that over reconstructing flags.

Tests (the repo's `setup.cfg` forces coverage addopts that break a bare run):

```bash
tools/anaconda/envs/espnet/bin/python -m pytest test/espnet2/fileio/test_indexed_text.py \
    -q -p no:cacheprovider -o addopts=""
```

---

## Settings that are load-bearing

Do not change these casually; each is a measured response to a specific failure.

**`batch_bins: 36000000`** — measured on an 8xH100 sweep (`local/bench/`). 48M is
a genuine OOM wall. Bins are approximately raw sample count.

**`num_iters_per_epoch: 40000`** — an "epoch" here is a checkpoint/validation
interval, not a corpus pass. A real pass is **314,333 steps**;
`max_epoch` alone would run for a year. `MultipleIterFactory` rebuilds the
dataset per split *inside* each epoch, so reload cost is fixed per epoch
regardless of length — at 5,000 iters, 76% of wall clock went to reloads.

**`--num_splits_ssl 8` AND `--lazy_km_labels true`** — both are required, and
neither is sufficient. They fix different terms of the same OOM. With lazy
labels but no splits, a rank still sat at 843 GB, because `wav.scp` and the two
shape files stay eager (~50 GB/rank at this corpus size). Lazy labels alone cut
the 147.9 GB label file from ~10 GB resident per shard to 0.04 GB.

**`--max_wav_duration 30.01`** — stage 4 compares with a strict `<`, not `<=`.
A cap of exactly 30 silently discards **81%** of this corpus.

**`use_torch_compile=true`** (in `run_large.sh`, which overrides the YAML) —
whole-model compile. See the compile section below.

**`unused_parameters: true`** — kept empirically, root cause unknown. See below.

**`num_att_plot: 0`** — pretraining has no attention plots worth having; the
reporter was spending 93 s per epoch to log `total_count=0`.

---

## Traps that have already cost time

**Releasing a multi-GPU allocation loses it.** Cancelling the job to change a
config freed 8 GPUs; within ~90 seconds someone took a single GPU on that node
(with a 46-day limit), and the 8-GPU request could not be satisfied anywhere.
Before any `scancel`, check `squeue` for pending GPU jobs and whether another
node can hold the allocation. Being *already queued* is the best defence — slurm
re-evaluates pending jobs every cycle.

**Slurm memory is bought via CPUs.** `MaxMemPerCPU = DefMemPerCPU = 14680M`, and
the gpu partition fixes 24 CPUs per GPU — `srun -c/--cpus-per-gpu` is rejected
outright. Lowering `--num_threads` lowers the memory ceiling and can cause an
OOM that looks unrelated.

**`expandable_segments` is NOT set.** A comment in `local/bench/sweep4.sh`
claims a python wrapper exports it. There is no such wrapper; the run uses the
default allocator. Peak is ~76-77 GB of 80.

**SIGPIPE under `set -o pipefail`.** `cmd | head` inside a command substitution
kills the driver silently. This bit twice, in `data_pretraining.sh` and in the
k-means duration guard. Both are now single-awk invocations. Do not reintroduce
a pipeline into a `$( )` in these scripts.

**Never compile submodules in place.** `mod[i] = torch.compile(mod[i])` mutates
the module tree, so the checkpointed `model.state_dict()` gets `_orig_mod.`
embedded in its keys (96% of 396 tensors), and `Trainer.resume` — which loads
`strict=True` and runs *before* compile — cannot load them.
`Trainer._maybe_compile` compiles the whole model and returns a new wrapper,
leaving `model` untouched. Keep it that way.

**`text_shape.word` contributes `L x vocab_size`** to `np.prod` in the numel
sampler, not `L`. Relevant if you change `n_clusters`.

---

## torch.compile: enabled, measured

Whole-model compile with `dynamic=True` (batches vary from 29 to 11,444
utterances; static shapes make dynamo recompile per shape and lose to eager).

| | eager | compiled |
|---|---|---|
| instantaneous | 0.603 s/step | 0.377-0.399 |
| **epoch average** | 0.603 | **0.436** |
| forward / backward | 0.212 / 0.295 | 0.152 / 0.188 |
| peak memory | 77.28 GB | 76.44 GB |

The epoch average is worse than the instantaneous rate because split reloads are
filesystem work that compile does not touch. **End-to-end gain is 28%**, not the
36% the live blocks suggest. First step pays ~3.6 s; a full pass drops from
52.6 h to ~38 h.

Historical note: two earlier compile attempts deadlocked, which is why the
committed YAML default is still `false` and the driver opts in. Those hangs are
now attributed to something other than compile — compile ran clean through the
batch that hung the job three times.

---

## The deadlock: fixed in practice, unexplained in theory

The run hung three times at **epoch 2, batch 13,100** — 100% GPU utilisation at
~145 W instead of ~490 W, no error, slurm still `RUNNING`. Setting
`unused_parameters: true` is the one change after which it ran through.

**It is not fixing an unused-parameter bug.** DDP prints, every run, that it
"did not find any unused parameters in the forward pass", and a direct census
over uniform, jagged and all-False-mask batches found **zero** parameters with
`grad=None`. The flag only perturbs DDP's bucketing. The underlying race is
probably still present and merely not triggering. It costs ~2.6%.

Hypotheses eliminated by experiment, so nobody repeats them:

| hypothesis | how it died |
|---|---|
| bad audio file | batch 28,300 contents unremarkable |
| compile / DDP graph order | hung in eager mode too; compile later ran clean |
| cross-job contention | hung while running alone |
| zero masked frames | see below |

The zero-mask theory was wrong and was written into config comments as fact
before being tested. Two errors: the call site
(`torchaudio .../wav2vec2/components.py:1040`) passes `min_masks=2`, so
`num_mask = max(2, ...)`; and a genuinely all-False mask does not deadlock, it
raises `ZeroDivisionError` on `correct_m / count_m` at
`espnet2/hubert/espnet_model.py:115`, which this run never threw (`count_m` ran
steady at 4,700-5,000).

There *is* a real quirk nearby, just not this one: `components.py:949-953`
truncates every utterance's mask to the **batch-wide minimum** span count, so one
degenerate row thins the mask for the whole batch.

**If it hangs again**, do not add another hypothesis — instrument the collective.
Log per-rank gradient-bucket ready order and find which rank stops arriving.

---

## Monitoring

The failure signature is a *silent* hang, so always include a liveness check —
absence of output is not health. A minimal watcher:

```bash
E=/mnt/weka/data/wavlm_expts/exp/wavlm_iter0_train_ssl_torchaudiowavlm_large_960h_pretrain_it0_raw
age=$(( $(date +%s) - $(stat -c %Y $E/train.log) ))   # alert if > 900
n=$(grep -ao "5epoch:train:[0-9]*-[0-9]*batch" $E/train.log | tail -1 | sed 's/.*-\([0-9]*\)batch/\1/')
```

Note the `sed`: block ranges are `4901-5000batch`, so parsing the *first* number
gives 4901, not 5000. Getting this wrong silently disables progress reporting.

A hang looks like: slurm `RUNNING`, GPUs at 100% utilisation but ~145 W, and
`train.log` not written for many minutes.

---

## Progress so far (iteration 0)

| epoch | train loss | train acc_m | valid loss | valid acc_m | notes |
|---|---|---|---|---|---|
| 1 | 1.237e+04 | 0.352 | 8.531e+03 | 0.445 | |
| 2 | 9.432e+03 | 0.458 | 7.875e+03 | 0.479 | best |
| 3 | 9.099e+03 | 0.475 | 7.531e+03 | 0.494 | best |
| 4 | 8.687e+03 | 0.487 | 7.552e+03 | 0.497 | first non-improving; compile on |
| 5 | in progress | | | | |

`acc_u` sits near 0.17-0.19 and is **not** optimised — `unmasked_weight` is
`0.0`, so it is a diagnostic readout, not a target. Chance is 0.01.

`patience: 4` is the real stopping rule, judged on `valid/loss`. Note the
off-by-one: `reporter.check_early_stopping` tests `epoch - best_epoch > patience`.

---

## Parked: continued pretraining (second track)

`run_large_ft.sh` — one iteration on top of torchaudio's pretrained WavLM-large
weights, rather than the 3-iteration from-scratch pipeline. Parked with its
artefacts intact:

- `exp/kmeans_iter2_espnet_wavlm_pretrain_train_portion0.0005/km_500.mdl` (2.0 MB, fit complete, inertia 13,804.9)
- ~52 GB of dumped GPU features (31,241 utts)

Resume from `perform_kmeans --stage 3` — about 33 minutes of labelling to redo.
Then `collect_stats`, which is **untested for this track** and may hit the same
`/dev/shm` failure that cost 3.5 h on iteration 0; the workaround is generating
shape files directly rather than via `collect_stats`.

It inherits the current compile and `unused_parameters` settings.

---

## Known-unfixed

- `load_feature_shard` reads **all** features before `--percent` samples them.
  Worked around with `portion_km 0.0005`, not fixed.
- K-means token-list sampling is systematically biased: 10x more data produced
  an identical rank error.
- `wav.scp` and the shape files are still eager. Making them lazy (the
  `IndexedTextReader` pattern in `espnet2/fileio/indexed_text.py` already exists
  and is tested) would allow dropping `--num_splits_ssl` and remove the ~17
  reload stalls per epoch — the largest remaining source of dead time.

---

## Conventions for whoever picks this up

Comments in the configs and scripts record *measurements and their provenance*,
not intentions — several of them exist specifically to stop a future reader
re-deriving a wrong conclusion. Two of those comments were themselves wrong and
had to be corrected after testing. If you assert a cause in a comment, say how
you verified it, and if you did not verify it, say that instead.
