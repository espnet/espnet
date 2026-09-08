# Speech Language Model

This is a template of speechlm1 recipe for ESPnet2.

`train.sh` launches `espnet2.speechlm.bin.train` from prepared dataset manifests
and length statistics. See the [Bagpiper](../../bagpiper/speechlm1/README.md) and
[Bagpiper-TTS](../../bagpiper_tts/speechlm1/README.md) examples for TorchTitan training.
The existing `speechlm.sh` driver uses `espnet2.bin.speechlm_train`.
