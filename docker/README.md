# ESPnet: end-to-end speech processing toolkit

Docker images from ESPnet https://github.com/espnet/espnet

See https://espnet.github.io/espnet/docker.html

### Tags

- Runtime: Base image for ESPnet. It includes libraries and Kaldi installation.
- CPU: Image to execute only in CPU.
- GPU: Image to execute examples with GPU support.

### Ubuntu 22.04

Python 3.10, Pytorch 2.6.0, No warp-ctc:

- [`cuda12.6` (*docker/prebuilt/gpu.dockerfile)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/gpu.dockerfile)
- [`cpu-u24` (*docker/prebuilt/devel.dockerfile)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel.dockerfile/Dockerfile)
