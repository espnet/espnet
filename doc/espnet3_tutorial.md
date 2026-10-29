# ESPnet3 tutorial

Read [the recipe migration guides](./espnet3/guides/index.html) for dataset,
training, inference and scaling workflows. Recipe `run.py` files call
`espnet3.systems.launch.launch`; systems declare `StageSpec` tuples rather than
maintaining a separate numbered-stage dispatcher.
