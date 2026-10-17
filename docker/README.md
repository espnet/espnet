# ESPnet: end-to-end speech processing toolkit

Docker images from ESPnet https://github.com/espnet/espnet

See https://espnet.github.io/espnet/docker.html

### Which image

Three images, for three different jobs. They are not versions of one another:
the development images exist to run recipes, the inference image exists to run
one published model on one file, and the vLLM image serves SpeechLM models.

| Image | What it is for | One command |
| :---- | :---- | :---- |
| `espnet/espnet:inference-cpu-latest` | Trying a published model - ASR, translation, TTS, enhancement - with nothing installed. | `docker run --rm -v "$PWD:/data" -v "$HOME/.cache/huggingface:/cache/huggingface" espnet/espnet:inference-cpu-latest asr /data/audio.wav` |
| `espnet/espnet:inference-gpu-latest` | The same, on a GPU: the CUDA 12.6 build of torch, for the 1B models where the CPU is the wait. | `docker run --rm --gpus all -v "$PWD:/data" -v "$HOME/.cache/huggingface:/cache/huggingface" espnet/espnet:inference-gpu-latest asr /data/audio.wav --device cuda` |
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
    -v "$HOME/.cache/huggingface:/cache/huggingface" \
    espnet/espnet:inference-cpu-latest asr /data/audio.wav
```

On a machine with a GPU, the same thing with the CUDA image. `--device cuda`
is what tells the command to use it; without it the model runs on the CPU
inside a much larger image, which is the worst of both:

```sh
docker run --rm --gpus all \
    -v "$PWD:/data" \
    -v "$HOME/.cache/huggingface:/cache/huggingface" \
    espnet/espnet:inference-gpu-latest asr /data/audio.wav --device cuda
```

The two images are built from one dockerfile and differ only in which torch
they install: `+cpu` against `+cu126`. The CUDA one is several gigabytes
larger, because the CUDA runtime travels inside those wheels - which is also
why it needs no CUDA base image, only a host driver new enough for CUDA 12.6.

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
- `-v "$HOME/.cache/huggingface:/cache/huggingface"` keeps the model
  between runs. No weights are baked into the image - one OWSM checkpoint is
  around 4 GB, and any tag from the [ESPnet Hugging Face
  organization](https://huggingface.co/espnet) can be asked for with `--model` -
  so the first run downloads one. Without this mount, so does every run after
  it. A model published outside the Hub is cached by `espnet_model_zoo` rather
  than by `huggingface_hub`, under `/cache/.cache/espnet_model_zoo`; mount that
  too if you use one.

On a Linux host, add `--user "$(id -u):$(id -g)"`. Without it the container
writes as root, and the transcript or wav it leaves in your directory is
root-owned. Everything the image writes - both caches - is under `/cache`,
which is world-writable for exactly this reason, so no other change is needed.
(On Docker Desktop for macOS and Windows the file sharing already maps
ownership, and the flag is unnecessary.)

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
