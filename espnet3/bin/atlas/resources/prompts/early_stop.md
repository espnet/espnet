Keypoint check for ${trial_id}: ${labels}.
Current progress: ${progress}.

Read current evidence with tools as needed; do not ask for pasted copies:
- Trial directory: ${trial_dir}
- Training log: ${trial_dir}/train.log
- Evaluation log: ${trial_dir}/eval.log
- Live training metrics: ${metrics_csv}
- Trial history and scores: ${study_dir}/trials.csv
- Study objective: ${study_dir}/program.md

Should this trial be early stopped?
Use the `pretrain_validation` entry in train.log as this trial's zero-shot baseline. At an epoch keypoint, stop if the current fixed-dev valid/loss is worse than that baseline; report both values in key_metrics.
Also stop if the trajectory is clearly inferior to other trials unlikely to recover.
Be conservative — prefer to continue unless the evidence is strong.
