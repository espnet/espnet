# ESPnet: end-to-end speech processing toolkit

Docker images from ESPnet https://github.com/espnet/espnet

See https://espnet.github.io/espnet/docker.html

### Tags

- Runtime: Base image for ESPnet. It includes libraries and Kaldi installation.
- CPU: Image to execute only in CPU.
- GPU: Image to execute examples with GPU support.

### Ubuntu 22.04

Python 3.9, Pytorch 1.13.1, No warp-ctc:

- [`cuda11.7` (*docker/prebuilt/gpu.dockerfile)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/gpu.dockerfile)
- [`cpu-u22` (*docker/prebuilt/devel.dockerfile)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel.dockerfile/Dockerfile)
