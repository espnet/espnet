# ESPnet3 ESPnet2 TSE template

This package supplies role defaults for `ESP2TSESystem`; it is not a runnable
recipe. Start with `egs3/librimix/esp2_tse` for LibriMix training or
`egs3/mini_an4/esp2_tse` for a small integration fixture.

Recipes call `launch(ESP2TSESystem)` and support `create_dataset`, `collect_stats`,
`train`, `infer`, `measure`, model/demo packing, and explicitly requested uploads.
Inference consumes mixture and target enrollment audio and writes `inf.scp` plus
WAVs. Scoring references come from `dataset:speech_ref1`, outside the model call.
The shared audio contract supplies the sample rate and validates mono inputs;
set both input/output contract rates and the WAV artifact rate when adapting
the 16 kHz defaults to another model. Batched TSE inference is not supported.
