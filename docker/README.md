# ESPnet: end-to-end speech processing toolkit

Docker images from ESPnet https://github.com/espnet/espnet

See https://espnet.github.io/espnet/docker.html

### Which image

Three images, for three different jobs. They are not versions of one another:
the development images exist to run recipes, the inference image exists to run
one published model on one file, and the vLLM image serves SpeechLM models.

| Image | What it is for | One command |
| :---- | :---- | :---- |
| `espnet/espnet:inference-latest` | Trying a published model - ASR, translation, TTS, enhancement - with nothing installed. CPU only. | `docker run --rm -v "$PWD:/data" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" espnet/espnet:inference-latest asr /data/audio.wav` |
| `espnet/espnet:cpu-latest`, `espnet/espnet:gpu-latest` | Development: running and writing `egs2` recipes, training, the Kaldi-style tooling. The whole toolkit. | `docker run --rm -it -v "$PWD:/work" espnet/espnet:cpu-latest bash` |
| `espnet/vllm:latest` | Serving a SpeechLM checkpoint behind an OpenAI-compatible API, on GPU. Built from the [ESPnet vLLM fork](https://github.com/espnet/vllm). | [Serving guide](https://github.com/espnet/espnet/pull/6695) (conversion, then `docker run --gpus all ... espnet/vllm:latest`) |

The `espnet/espnet:*` tags are built weekly from this directory by
[`.github/workflows/publish_docker_image.yml`](../.github/workflows/publish_docker_image.yml).
`espnet/vllm` is built from the fork, not from here.

### Inference image

Its entry point is the `espnet` command that `pip install espnet` ships, so
every subcommand of that command is an argument to `docker run`:

```sh
docker run --rm \
    -v "$PWD:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    espnet/espnet:inference-latest asr /data/audio.wav
```

With the same two mounts, the other subcommands are:

```text
asr /data/audio.wav                     transcribe, detecting the language
translate /data/audio.wav --to eng      speech in, English text out
tts "Hello from ESPnet" -o /data/out.wav
enhance /data/noisy.wav -o /data/clean.wav
models                                  the default model of each command
```

Both mounts matter:

- `-v "$PWD:/data"` is how the container sees your audio, and where a file
  written with `-o` ends up. The working directory inside the image is `/data`,
  so that is also what a relative path means.
- `-v "$HOME/.cache/huggingface:/root/.cache/huggingface"` keeps the model
  between runs. No weights are baked into the image - one OWSM checkpoint is
  around 4 GB, and any tag from the [ESPnet Hugging Face
  organization](https://huggingface.co/espnet) can be asked for with `--model` -
  so the first run downloads one. Without this mount, so does every run after
  it. A model published outside the Hub is cached in `~/.cache/espnet_model_zoo`
  instead, so mount that as `/root/.cache/espnet_model_zoo` if you use one.

`--model <tag>` selects any published model that suits the subcommand;
`espnet models` names the default of each. The image is CPU-only - it installs
the `+cpu` build of torch, so `--device cuda` has nothing to talk to. For GPU
inference use the development GPU image, or `pip install espnet` in a CUDA
environment.

To build it locally, or to build one for a release other than the current one:

```sh
docker build -f inference.dockerfile --build-arg ESPNET_VERSION=202610 -t espnet-inference .
```

### Development images

Built on the prebuilt bases in [`prebuilt/`](prebuilt). `run.sh` wraps
`docker run` for recipes:

```sh
./run.sh --docker-gpu 0 --docker-egs an4/asr1 --docker-cmd run.sh
```

### Tags

- Runtime: Base image for ESPnet. It includes libraries and Kaldi installation.
- CPU: Image to execute only in CPU.
- GPU: Image to execute examples with GPU support.
- Inference: `pip install espnet` and nothing else, to run published models.

### Ubuntu 22.04

Python 3.12, Pytorch 2.11.0, No warp-ctc:

- [`cuda12.6` (*docker/prebuilt/gpu.dockerfile)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/gpu.dockerfile)
- [`cpu-u24` (*docker/prebuilt/devel.dockerfile)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel.dockerfile/Dockerfile)
