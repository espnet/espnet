# Speech Language Model

This template provides `speechlm.sh` for the staged ESPnet SpeechLM interface,
using `espnet2.bin.speechlm_train`. GigaSpeech, LibriSpeech, LibriTTS, and mini_an4
recipes link to its driver and supporting files.

Use `setup.sh <target-dir>` to scaffold a recipe with the traditional ESPnet
environment, data-preparation stages, and job launcher. The script copies
`cmd.sh`, `conf/`, and `local/`, and links `speechlm.sh`, `path.sh`, `db.sh`,
`scripts`, `pyscripts`, `steps`, and `utils` into the target directory.

[Bagpiper](../../bagpiper/speechlm1/README.md) and
[Bagpiper-TTS](../../bagpiper_tts/speechlm1/README.md) use a separate training
interface, `espnet2.speechlm.bin.train`. Each recipe's `run.sh` launches TorchTitan
training directly from prepared dataset manifests and length statistics.
See those recipe READMEs for environment setup, launch options, and checkpoint use.
