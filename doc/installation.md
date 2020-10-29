## Installation
### Requirements

- Python 3.6.1+
- gcc 4.9+ for PyTorch1.0.0+

Optionally, GPU environment requires the following libraries:

- Cuda 8.0, 9.0, 9.1, 10.0 depending on each DNN library
- Cudnn 6+, 7+
- NCCL 2.0+ (for the use of multi-GPUs)

(If you'll use anaconda environment at installation step2,
the following packages are installed using Anaconda, so you can skip them.)

- cmake3 for some extensions
    ```sh
    # For Ubuntu
    $ sudo apt-get install cmake
    ```
- sox
    ```sh
    # For Ubuntu
    $ sudo apt-get install sox
    # For CentOS
    $ sudo yum install sox
    ```
- sndfile
    ```sh
    # For Ubuntu
    $ sudo apt-get install libsndfile1-dev
    # For CentOS
    $ sudo yum install libsndfile
    ```
- ffmpeg (This is not required when installataion, but used in some recipes)
    ```sh
    # For Ubuntu
    $ sudo apt-get install ffmpeg
    # For CentOS
    $ sudo yum install ffmpeg
    ```
- flac (This is not required when installataion, but used in some recipes)
    ```sh
    # For Ubuntu
    $ sudo apt-get install flac
    # For CentOS
    $ sudo yum install flac
    ```

### Supported Linux distributions and other requirements

We support the following Linux distributions with CI. If you want to build your own Linux by yourself,
please also check our [CI configurations](https://github.com/espnet/espnet/blob/master/.circleci/config.yml).
to prepare the appropriate environments

- ubuntu18
- ubuntu16
- centos7
- debian9


### Step 1) Install Kaldi
Related links:
- [Kaldi Github](https://github.com/kaldi-asr/kaldi)
- [Kaldi Documentation](https://kaldi-asr.org/)
  - [Downloading and installing Kaldi](https://kaldi-asr.org/doc/install.html)
  - [The build process (how Kaldi is compiled)](https://kaldi-asr.org/doc/build_setup.html)
- [Kaldi INSTALL](https://github.com/kaldi-asr/kaldi/blob/master/INSTALL)

Kaldi's requirements:
- OS: Ubuntu, CentOS, MacOSX, Windows, Cygwin, etc.
- GCC >= 4.7

1. Git clone Kaldi

    ```sh
    $ cd <any-place>
    $ git clone https://github.com/kaldi-asr/kaldi
    ```
1. Install tools

    ```sh
    $ cd <kaldi-root>/tools
    $ make -j <NUM-CPU>
    ```
    1. Select BLAS library from ATLAS, OpenBLAS, or MKL

    - OpenBLAS

    ```sh
    $ cd <kaldi-root>/tools
    $ ./extras/install_openblas.sh
    ```
    - MKL (You need sudo privilege)

    ```sh
    $ cd <kaldi-root>/tools
    $ sudo ./extras/install_mkl.sh
    ```
    - ATLAS (You need sudo privilege)

    ```sh
    # Ubuntu
    $ sudo apt-get install libatlas-base-dev
    ```

1. Compile Kaldi & install

    ```sh
    $ cd <kaldi-root>/src
    # [By default MKL is used] ESPnet uses only feature extractor, so you can disable CUDA
    $ ./configure --use-cuda=no
    # [With OpenBLAS]
    # $ ./configure --openblas-root=../tools/OpenBLAS/install --use-cuda=no
    # If you'll use CUDA
    # ./configure --cudatk-dir=/usr/local/cuda-10.0
    $ make -j clean depend; make -j <NUM-CPU>
    ```
We also have [prebuilt Kaldi binaries](https://github.com/espnet/espnet/blob/master/ci/install_kaldi.sh).

### Step 2) Installation ESPnet
1. Git clone ESPnet
    ```sh
    $ cd <any-place>
    $ git clone https://github.com/espnet/espnet
    ```
1. Put Kaldi at espnet/tools

    Create a symbolic link to Kaldi directory.

    ```sh
    $ cd <espnet-root>/tools
    $ ln -s <kaldi-root> .
    ```
1. Setup CUDA environment

    Specify your CUDA directory.

    ```sh
    $ cd <espnet-root>/tools
    $ . ./setup_cuda_env.sh <cuda-root>  # e.g. <cuda-root> = /usr/local/cuda
    # If you have NCCL (If you'll install pytorch from anaconda, NCCL is also bundled, so you don't need to give it)
    # $ . ./setup_cuda_env.sh <cuda-root> <nccl-root> # e.g. <nccl-root> = /usr/local/nccl
    ```
1. Setup Python environment

    The Python interpreter used in espnet recipes is determined by `<espnet-root>/tools/activate_python.sh`,
    and in this step, you need to create the file.

    You must select one of setup scripts for Python environment here.
    We prepare scripts for `Anaconda`, `venv`, and `System Python` environment, so
    if you'll use the other Python environment manager, e.g. `pipenv`, `pyenv`, or etc.
    you need to create `activate_python.sh` by yourself.

    If you don't stick to any Python environments, please select `Anaconda` environment.

    - Option A) Setup Anaconda environment

        ```sh
        $ cd <espnet-root>/tools
        $ ./setup_anaconda.sh [output-dir-name] [conda-env-name] [python-version]
        ```
        If `[output-dir-name]` is omitted, `venv` is generated.
        If `[conda-env-name]` and `[python-version]` are omitted,
        the root environment and the default Python of Anaconda are selected respectively.

        This script tries to create a new miniconda at `venv` if it doesn't exist.
        If you already have Anaconda and you'll use it then,

        ```sh
        $ cd <espnet-root>/tools
        $ CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..
        $ ./setup_anaconda.sh ${CONDA_TOOLS_DIR} [conda-env-name] [python-version]
        ```

    - Option B) Setup venv from system Python

        ```sh
        $ cd <espnet-root>/tools
        $ ./setup_venv.sh $(command -v python3)
        ```

    - Option C) Setup system Python environment

        ```sh
        $ cd <espnet-root>/tools
        $ ./setup_python.sh $(command -v python3)
        ```
1. Install ESPnet

    ```sh
    $ cd <espnet-root>/tools
    $ make
    ```

    The Makefile tries to install ESPnet and all dependencies including PyTorch.
    You can also specify PyTorch version, for example:

    ```sh
    $ cd <espnet-root>/tools
    $ make TH_VERSION=1.3.1
    ```

    If you don't have `nvcc` command, packages are installed for CPU mode by default.
    If you'll turn it on manually, give `CPU_ONLY` option.

    ```sh
    $ cd <espnet-root>/tools
    $ make CPU_ONLY=0
    ```
### Step 3) [Option] Manual installation
If you are stuck in some troubles when installation, you can also install them ignoring the Makefile.

Note that the Python interpreter used in ESPnet experiments is written in `<espnet-root>/tools/activate_python.sh`,
so you need to activate it before installing python packages.

```sh
cd <espnet-root>/tools
. activate_python.sh
python3 -m pip install <some-package>
./installers/install_<some-tool>.sh
```

### Check installation
You can check whether your installation is succesfully finished by
```sh
cd <espnet-root>/tools
. ./activate_python.sh; python3 check_install.py
```
Note that this check is always called in the last stage of the above installation.
