---
name: autoresearch-log-analysis
description: Inspect AutoResearch training and evaluation logs or metric curves when proposing a trial or debugging a failed run.
---

# AutoResearch log analysis

Use the log paths supplied in the task context. Read files on demand; do not copy whole logs into the response.

- For failures, inspect the current attempt's `stdout.log` and `stderr.log` first, then the relevant tail of `train.log` or `eval.log`. Search for tracebacks, errors, and the first failure before suggesting a fix.
- For trial proposals, inspect `training_metrics.csv` and relevant `exp/csv_logs/**/metrics.csv` files from the supplied trial directories. Compare progress and final metrics across a few recent trials.
- Use targeted shell commands such as `rg -n -i 'error|exception|traceback|failed|out of memory' FILE` and `tail -n 120 FILE`. For CSV data, select the header and recent rows instead of printing a large file.
- Ground conclusions in file paths and specific log lines or metric values. State when a referenced file is missing or contains no useful data.
- Do not launch training or modify logs while collecting evidence.
