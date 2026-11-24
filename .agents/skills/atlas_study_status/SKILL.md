---
name: atlas_study_status
description: Check whether an Atlas study (hyperparameter or architecture search) is healthy and how far it has got, without depending on a particular job scheduler.
---

# Check an Atlas study

Use this to answer "is the study running?", "how far are the trials?", or "is the controller alive?". Work from recorded state first, and touch the job scheduler only for the last step.

Replace `<study_dir>` with the study directory (`atlas.study_dir` in the recipe's `atlas.yaml`, resolved against the recipe directory).

## 1. Summary from recorded state

```bash
espnet3 atlas status --study-dir <study_dir>
```

Read it as follows. It reads only the study's state files and trial logs and never asks the scheduler, so a recorded job status can lag behind reality.

- `trials`, `best`, `trial_status`: study progress against `atlas.budget.max_trials`.
- `stop_file: yes`: a `STOP` file exists, so the controller will not start new work. Remove it only if the user wants the study to continue.
- `controller_last_state_update`: age of the last recorded state change. The controller ticks every `tick_interval_sec_trial_active` seconds while trials run (default 600) and `tick_interval_sec_idle` when idle. A value several times larger than that interval suggests the controller is not running. A longer gap is normal if nothing changed, so confirm with step 2.
- `active_jobs`: jobs recorded as unfinished. A `tick` job stuck as `submitted` or `running` after the controller died is a stale record; do not treat it as a live controller.
- `running_trial_progress`: `epoch X/Y` per running trial, from each trial's `exp/csv_logs/**/metrics.csv`.

## 2. Is a controller process alive?

A controller is either a process started with `espnet3 atlas run ... --worker` (for example in tmux on a login node) or a submitted `tick` job.

```bash
pgrep -af "atlas run"
```

If there is no process and no live tick job, the controller is down. Running trial jobs keep going, but no new trial is started and results are only recorded when a trial job's own self-controller step runs (`gpu_self_controller.enabled`).

## 3. Are the recorded jobs really alive?

Use the scheduler named by `atlas.submission.environment` and the commands in `atlas.submission.instructions` (for example `qstat` for PBS, `squeue`/`sacct` for Slurm, `ps` for local). Compare against the `id=` values from `active_jobs`. Report any job the scheduler no longer lists as finished or dead, and say that the study's record is stale.

## 4. Look at a trial that failed or looks stuck

- `<study_dir>/trials/<trial_id>/train.log` and `attempts/attempt_*/` (scheduler stdout/stderr).
- `<study_dir>/trials/<trial_id>/debug.md`, written by the debug stage when a retry was planned.
- `<study_dir>/state_events.csv` for the sequence of scheduling events.

## Rules

- This skill is read-only. Do not delete files, cancel jobs, create or remove `STOP`, or submit jobs unless the user asks.
- Report facts with the file or command they came from, and say when a conclusion is an inference (for example "controller probably down").
