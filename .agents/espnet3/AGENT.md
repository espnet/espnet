# `espnet3/`

This is a navigation point for the shared package guides, not a copy of the
source tree. Use `rg --files espnet3` or inspect [`espnet3/`](../../espnet3/)
for the current module layout.

- [Inference](inference/AGENT.md): declared model inputs and outputs, the
  runner, and loading packed bundles.
- [Components](components/AGENT.md): data, modeling, training, metrics, and
  optimization building blocks.
- [`contract/`](../../espnet3/contract/): checks a part against its own
  declaration at run time (a metric's inputs/outputs, a dataset's item
  fields, a system's stage order, the inference inputs/outputs).
- [Parallel execution](parallel/AGENT.md): runners, providers, and worker
  boundaries.
- [Publication](publication/AGENT.md): model bundles and demo packaging.
- [Systems](systems/AGENT.md): stage orchestration and task-family boundaries.
- [Utilities](utils/AGENT.md): shared configuration, logging, and helpers.

Read the implementation and tests named by the relevant guide before relying
on details. Keep cross-package behavior in shared abstractions and preserve
public contracts across their callers.
