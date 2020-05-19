# FAQ
## ModuleNotFoundError: No module named 'espnet', 'warpctc_pytorch', or etc.

Firstly, you definitely missed some installation processes. Please read [Installation](./installation.md) again before posting an issue. If you still have a problem, then you have to understand the following items.

### How does ESPnet depend on a python interpreter?
1. `egs/<some-where>/<some-task>/run.sh`, and any other shell scripts, always invokes `egs/<some-where>/<some-task>/path.sh` 
1. `path.sh` setups the python interpreter at `tools/venv`
1. `tools/venv` is installed by `tools/Makefile`

Note that **we normally don't use the system Python, but uses `tools/venv` installed by Makefile**.
See [Step 3-A) installation of espnet](installation.md#step-3-a-installation-of-espnet). We intend to prepare a Python interpreter independently and we'd also like to avoid contaminating the other Python. We believe this is a good strategy for a researcher, but sometimes it causes an installation problem. 

Please also note that, if you just intend to use Python scripts directly without a shell script, e.g. `asr_recog.py`, you always need to activate the Python interpreter before using it.

```bash
$ cd egs/<some-where>/<some-task>
$ source path.sh
```

### To detect the installation problem with a normal installation

1. Check where your python is
    ```bash
    $ cd egs/an4/asr1/
    $ source path.sh  # Activate the Python environment
    $ which python  # Normally, it should point to <espnet-root>/tools/venv
    ```
1. Check the installation of espnet
    ```bash
    $ python
    >>> import espnet
    >>> import warpctc_pytorch   # If you'll use warpctc
    ```

If you meet an error, redo [Installation](./installation.md).
