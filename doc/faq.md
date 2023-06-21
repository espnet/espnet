# FAQ
## How to build espnet on a cloud machine such as GCP, AWS, etc.?

Our documentation, [Installation](./installation.md), assumes that some basic tools are already installed in your machine, gcc, make, etc., 
so you need to also install them if you don't have. They are undocumented, but the configuration of our CI may help you because it also build the environment from scratch:  https://github.com/espnet/espnet/blob/master/.circleci/config.yml


## ModuleNotFoundError: No module named 'espnet', or etc.

Firstly, you definitely missed some installation processes. Please read [Installation](./installation.md) again before posting an issue. If you still have a problem, then please try to manual installation.

```sh
. tools/activate_python.sh
pip install <some-tools>
conda install <some-tools>
```

If you need to install some packages not distributed in pypi, e.g. `k2`, try to use the installer scripts in espnet.

```
cd tools
./installers/install_warp-transducer.sh
```

### To detect the installation problem with a normal installation

1. Check where your python is
    ```bash
    $ . tools/activate_python.sh
    $ which python  # Normally, it should point to <espnet-root>/tools/venv
    ```
1. Check the installation of espnet
    ```bash
    $ python
    >>> import espnet
    ```
