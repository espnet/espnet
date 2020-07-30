## Installation
### Requirements

- Python 3.6.1+
- gcc 4.9+ for PyTorch1.0.0+
- cmake3 for some extensions
    ```sh
    # e.g. Ubuntu
    $ sudo apt-get install cmake
    # e.g. Using anaconda (If you don't have sudo privilege, the installation from conda might be useful)
    $ conda install cmake
    ```

We often use audio converter tools in several recipes:

- sox
    ```sh
    # e.g. Ubuntu
    $ sudo apt-get install sox
    # e.g. CentOS
    $ sudo yum install sox
    # e.g. Using anaconda
    $ conda install -c conda-forge sox
    ```
- sndfile
    ```sh
    # e.g. Ubuntu
    $ sudo apt-get install libsndfile1-dev
    # e.g. CentOS
    $ sudo yum install libsndfile
    ```
- ffmpeg (This is not required when installataion, but used in some recipes)
    ```sh
    # e.g. Ubuntu
    $ sudo apt-get install ffmpeg
    # e.g. CentOS
    $ sudo yum install ffmpeg
    # e.g. Using anaconda
    $ conda install -c conda-forge ffmpeg
    ```
- flac (This is not required when installataion, but used in some recipes)
    ```sh
    # e.g. Ubuntu
    $ sudo apt-get install flac
    # e.g. CentOS
    $ sudo yum install flac
    ```

Optionally, GPU environment requires the following libraries:

- Cuda 8.0, 9.0, 9.1, 10.0 depending on each DNN library
- Cudnn 6+, 7+
- NCCL 2.0+ (for the use of multi-GPUs)

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

We also have [prebuilt Kaldi binaries](https://github.com/espnet/espnet/blob/master/ci/install_kaldi.sh).


1. Git clone kaldi

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
    # e.g. With OpenBLAS]
    # $ ./configure --openblas-root=../tools/OpenBLAS/install --use-cuda=no
    # If you'll use CUDA
    # ./configure --cudatk-dir=/usr/local/cuda-10.0
    $ make -j clean depend; make -j <NUM-CPU>
    ```

### Step 2) installation of espnet

```sh
$ cd <any-place>
$ git clone https://github.com/espnet/espnet
```

Before installing ESPnet, set up some environment variables related to CUDA.

```sh
$ cd <espnet-root>/tools
$ . ./setup_cuda_env.sh /usr/local/cuda
# If you have NCCL (If you'll install pytorch from anaconda, NCCL is also bundled, so you don't need to give it)
# $ . ./setup_cuda_env.sh /usr/local/cuda /usr/local/nccl
```

#### Option A) create miniconda environment (default)
```sh
$ cd <espnet-root>/tools
$ make KALDI=<kaldi-root>
```

You can also specify the Python and PyTorch version, for example:
```sh
$ cd <espnet-root>/tools
$ make KALDI=<kaldi-root> PYTHON_VERSION=3.6 TH_VERSION=1.3.1
```

By default, the environment is named `espnet`. If you prefer the other name,

```sh
$ cd <espnet-root>/tools
$ make KALDI=<kaldi-root> CONDA_ENV_NAME=<name>
```

#### Option B) create environment in existing anaconda/change the installation path of anaconda

If you already have anaconda and you'll create an environment of ESPnet.

```sh
$ cd <espnet-root>/tools
$ CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..
$ make KALDI=<kaldi-root> CONDA=${CONDA_TOOLS_DIR} CONDA_ENV_NAME=<name>
```

Before executing this command, check the existance of `${CONDA_TOOLS_DIR}/etc/profile.d/conda.sh`

Note that
- If there are no conda tools at the path, new conda is created there.
- If there already exists conda and its environment, the creation of a new environment is skipped.

#### Option C) create virtualenv from an existing python

If you do not want to use miniconda, you need to specify your python interpreter to setup `virtualenv`

```sh
$ cd <espnet-root>/tools
$ make KALDI=<kaldi-root> PYTHON=/usr/bin/python3.6
```

In this case, you can't use `PYTHON_VERSION` option, but you can still use `TH_VERSION`.


```sh
$ cd <espnet-root>/tools
$ make KALDI=<kaldi-root> PYTHON=/usr/bin/python3.6 TH_VERSION=1.3.1
```

#### Option D) using the existing Python/Anaconda without creating new environment
You can skip the installation of new environment by creating `activate_python.sh`.

```sh
$ cd <espnet-root>/tools
$ rm -f activate_python.sh; touch activate_python.sh
$ make KALDI=<kaldi-root> USE_PIP=0
```

If your Python is anaconda, you don't need to provide `USE_PIP`.

```sh
$ cd <espnet-root>/tools
$ rm -f activate_python.sh; touch activate_python.sh
$ make KALDI=<kaldi-root>
```

In this case, you can use `TH_VERSION` too.

Note that we don't append `--user` option when pip instllation, so you must have write privilege to your Python.

#### Option E) installation for CPU-only

If you don't have `nvcc` command, packages are installed for CPU mode by default.
If you'll turn it on manually, give `CPU_ONLY` option.

```sh
$ cd <espnet-root>/tools
$ make KALDI=<kaldi-root> CPU_ONLY=0
```

This option is enabled for any of the install configuration.


### Step 3) installation check
You can check whether the install is succeeded via the following commands

```sh
$ cd <espnet-root>/tools
$ make check_install
```

If there are some problems in python libraries, you can re-setup only python environment via following commands
```sh
$ cd <espnet-root>/tools
$ make clean_python
$ make python
```

### Step 4) [Option] Manual installation
If you are stuck in some troubles when installation, you can also install them ignoring the Makefile.

Note that the Python interpreter used in ESPnet experiments is written in `<espnet-root>/tools/activate_python.sh`,
so you need to activate it before installing python packages.

```sh
cd <espnet-root>/tools
. activate_python.sh
pip3 install <some-package>
./installers/install_<some-tool>.sh
```
