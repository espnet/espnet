# ESPnet: end-to-end speech processing toolkit

Docker images from ESPnet https://github.com/espnet/espnet [![Build Status](https://travis-ci.org/espnet/espnet.svg?branch=master)](https://travis-ci.org/espnet/espnet)

## How to use 

To work inside a docker container, execute `run.sh` located inside the docker directory.
It will download the requested image and build a container to execute the main program specified by the following GPU, ASR example, and outside directory information, as follows:
```sh
$ cd docker
$ ./run.sh --docker_gpu 0 --docker_egs chime4/asr1 --docker_folders /export/corpora4/CHiME4/CHiME3 --dlayers 1 --ngpu 1 
```
Optionally, you can set the CUDA version with the arguments `--docker_cuda` respectively (default version set at CUDA=9.1). The docker container can be built based on the CUDA installed in your computer if you empty this arguments.
By default, all GPU-based images are built with NCCL v2 and CUDNN v7. 
The arguments required for the docker configuration have a prefix "--docker" (e.g., `--docker_user`, `--docker_gpu`, `--docker_egs`, `--docker_folders`). `run.sh` accept all normal ESPnet arguments, which must be followed by these docker arguments.
All docker containers are executed using the same user as your login account. If you want to run the docker in root access, set the `--docker_user` to `false`. In addition, you can pass any enviroment variable using `--docker_env` (e.g., `--docker_env "foo=path"`)
Multiple GPUs should be specified with the following options:
```sh
$ cd docker
$ ./run.sh --docker_gpu 0,1,2 --docker_egs chime5/asr1 --docker_folders /export/corpora4/CHiME5 --ngpu 3
```
Note that all experimental files and results are created under the normal example directories (`egs/<example>/`).

Multiple folders and environment variables should be specified with commas and without spaces:
```sh
$ cd docker
$ ./run.sh --docker_gpu 0 --docker_egs chime4/asr1 --docker_folders /export/corpus/CHiME4,/export/corpus/LDC/LDC93S6B,/export/corpus/LDC/LDC94S13B --docker_env "CHIME4_CORPUS=/export/corpus/CHiME4/CHiME3,WSJ0_CORPUS=/export/corpus/LDC/LDC93S6B,WSJ1_CORPUS=/export/corpus/LDC/LDC94S13B" --ngpu 1
```

## Tags

- Runtime: Base image for ESPnet. It includes libraries and Kaldi installation.
- CPU: Image to execute only in CPU. 
- GPU: Image to execute examples with GPU support.

# Ubuntu 16.04 

- [`cuda10.0-cudnn7` (*docker/prebuilt/gpu/10.0/cudnn7/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel/gpu/10.0/cudnn7/Dockerfile)
- [`cuda9.2-cudnn7` (*docker/prebuilt/gpu/9.2/cudnn7/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel/gpu/9.2/cudnn7/Dockerfile)
- [`cuda9.1-cudnn7` (*docker/prebuilt/gpu/9.1/cudnn7/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel/gpu/9.1/cudnn7/Dockerfile)
- [`cuda9.0-cudnn7` (*docker/prebuilt/gpu/9.0/cudnn7/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel/gpu/9.0/cudnn7/Dockerfile)
- [`cuda8.0-cudnn7`(*docker/prebuilt/gpu/8.0/cudnn7/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel/gpu/8.0/cudnn7/Dockerfile)
- [`cpu` (*docker/prebuilt/devel/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel/Dockerfile)
- [`runtime` (*docker/prebuilt/runtime/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/runtime/Dockerfile)