# Tools for setup

See also https://espnet.github.io/espnet/installation.html

## Contents
```
installers/        # Instllation scripts for extra tools
Makefile           # Makefile to make an environment for experiments in espnet
check_install.py   # To check the status of installation 
extra_path.sh      # Setup script for environment variables for extra tools
setup_cuda_env.sh  # Setup script for environment variables for cuda
setup_anaconda.sh  # To generate activate_python.sh with conda environment
setup_python.sh    # To generate activate_python.sh with the specified python
setup_venv.sh      # To generate activate_python.sh with venv of your python
```

## Check installation

```
. ./activate_python.sh; . ./extra_path.sh; python3 check_install.py
```
