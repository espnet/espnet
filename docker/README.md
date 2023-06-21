# ESPnet: end-to-end speech processing toolkit

Docker images from ESPnet https://github.com/espnet/espnet [![Build Status](https://travis-ci.org/espnet/espnet.svg?branch=master)](https://travis-ci.org/espnet/espnet)


See https://espnet.github.io/espnet/docker.html

### Tags

- Runtime: Base image for ESPnet. It includes libraries and Kaldi installation.
- CPU: Image to execute only in CPU.
- GPU: Image to execute examples with GPU support.

### Ubuntu 18.04

Pytorch 1.3.1, No warp-ctc:

- [`cuda10.1-cudnn7` (*docker/prebuilt/gpu/10.1/cudnn7/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel/gpu/10.1/cudnn7/Dockerfile)

Pytorch 1.0.1, warp-ctc:

- [`cuda10.0-cudnn7` (*docker/prebuilt/gpu/10.0/cudnn7/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel/gpu/10.0/cudnn7/Dockerfile)
- [`cpu-u18` (*docker/prebuilt/devel/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel/Dockerfile)
