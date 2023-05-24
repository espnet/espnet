---
name: Installation issue template
about: Create a report for installation issues
title: ''
labels: Installation
assignees: ''

---

**Describe the issue**
A clear and concise description of what the issue is.
Please check https://espnet.github.io/espnet/installation.html in advance.

**Show the `check_install.py` status by using the following command**
```
cd <espnet-root>/tools
. ./activate_python.sh; python3 check_install.py
```

**Basic environments:**
 - OS information: [e.g., Linux 4.9.0-11-amd64 #1 SMP Debian 4.9.189-3+deb9u2 (2019-11-11) x86_64]
 - python version: [e.g. 3.7.3 (default, Mar 27 2019, 22:11:17) [GCC 7.3.0]]
 - espnet version: [e.g. espnet 0.8.0]
 - Git hash [e.g. b88e89fc7246fed4c2842b55baba884fe1b4ecc2]
   - Commit date [e.g. Tue Sep 1 09:32:54 2020 -0400]
 - pytorch version [e.g. pytorch 1.4.0]

You can obtain them by the following command
```
cd <espnet-root>/tools
. ./activate_python.sh

echo "- OS information: `uname -mrsv`"
python3 << EOF
import sys, espnet, torch
pyversion = sys.version.replace('\n', ' ')
print(f"""- python version: \`{pyversion}\`
- espnet version: \`espnet {espnet.__version__}\`
- pytorch version: \`pytorch {torch.__version__}\`""")
EOF
cat << EOF
- Git hash: \`$(git rev-parse HEAD)\`
  - Commit date: \`$(git log -1 --format='%cd')\`
EOF
```

**Environments from `torch.utils.collect_env`:**
e.g.,
```
Collecting environment information...
PyTorch version: 1.4.0
Is debug build: No
CUDA used to build PyTorch: 10.0

OS: CentOS Linux release 7.5.1804 (Core)
GCC version: (GCC) 7.2.0
CMake version: version 2.8.12.2

Python version: 3.7
Is CUDA available: Yes
CUDA runtime version: 10.0.130
GPU models and configuration:
GPU 0: TITAN RTX
GPU 1: TITAN RTX
GPU 2: TITAN RTX
GPU 3: TITAN RTX

Nvidia driver version: 440.33.01
cuDNN version: Could not collect

Versions of relevant libraries:
[pip3] numpy==1.18.5
[pip3] pytorch-ranger==0.1.1
[pip3] pytorch-wpe==0.0.0
[pip3] torch==1.4.0
[pip3] torch-complex==0.1.1
[pip3] torch-optimizer==0.0.1a14
[pip3] torchaudio==0.4.0
[pip3] warprnnt-pytorch==0.1
[conda] blas                      1.0                         mkl
[conda] mkl                       2020.1                      217
[conda] mkl-service               2.3.0            py37he904b0f_0
[conda] mkl_fft                   1.1.0            py37h23d657b_0
[conda] mkl_random                1.1.1            py37h0573a6f_0
[conda] pytorch                   1.4.0           py3.7_cuda10.0.130_cudnn7.6.3_0    pytorch
[conda] pytorch-ranger            0.1.1                    pypi_0    pypi
[conda] pytorch-wpe               0.0.0                    pypi_0    pypi
[conda] torch-complex             0.1.1                    pypi_0    pypi
[conda] torch-optimizer           0.0.1a14                 pypi_0    pypi
[conda] torchaudio                0.4.0                    pypi_0    pypi
[conda] warprnnt-pytorch          0.1                      pypi_0    pypi
```
You can obtain them by the following command

```
cd <espnet-root>/tools
. ./activate_python.sh
python3 -m torch.utils.collect_env
```

**To Reproduce**
Steps to reproduce the behavior by showing us the specific installation commands with their arguments, e.g.,
```
cd <espnet-root>/tools
make TH_VERSION=1.3.1
```

**Error logs**
Paste the error logs. If applicable, add screenshots to help explain your problem.
