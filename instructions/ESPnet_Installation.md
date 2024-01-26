# 1. Install CUDA libraries:
( NOT REQUIRED WHEN RUNNING ON GOOGLE COLAB )

Check if nvcc is installed.
```bash
conda deactivate
cd ~
nvcc --version
```
If not installed, install by running
```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
nvcc --version
sudo reboot
```

# 2. Configure git to clone from private fork:
( OPTIONAL STEP )
```bash
git config user.name ThishRaj
git config user.email thishraj.96@gmail.com
git
ssh-keygen -t rsa -b 4096 -C thishraj.96@gmail.com
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub
```
Copy and paste the text that appears after running the command in `www.github.com > Your profile > Settings > SSH and GPG keys > New SSH Key / Add SSH Key.`
Setup a configuration file to configure the port number for connecting to github:
```bash
nano ~/.ssh/config
  Host github.com
    Hostname ssh.github.com
    Port 443
ssh -T git@github.com
```
# 3. Download ESPnet and Kaldi
- Git clone ESPnet 
```bash
cd ~
git clone git@github.com:ThishRaj/espnet.git
```
- Download Kaldi:
```bash
git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
```
- Link downloaded Kaldi to ESPnet
```bash
cd <espnet-root>/tools
ln -s <kaldi-root>
```      

# 4. Setup Python environment using conda.
This script tries to create a new miniconda if the output directory doesn’t exist. If you already have conda and you’ll use it, then,
```bash
cd <espnet-root>/tools
conda activate
CONDA_ROOT=${CONDA_PREFIX}  # CONDA_PREFIX is an environment variable set by ${CONDA_ROOT}/etc/profile.d/conda.sh
./setup_anaconda.sh ${CONDA_ROOT} [conda-env-name] [python-version]
```
Example:
```bash
./setup_anaconda.sh ${CONDA_ROOT} espnet 3.9
```      

# 5. Install ESPnet
Ensure cuda version while running espnet is same as that displayed by `nvidia-smi`.
```bash
conda activate espnet
cd <espnet-root>/tools
make TH_VERSION=1.13.1 CUDA_VERSION=11.8
```
# 6. Install other tools:
1. **`kenlm`**

Install `kenlm` for statistical LM training:
```bash
cd <espnet-root>/tools
./installers/install_kenlm.sh
```

2. **`warp-transducer`**

Install `warp-transducer` for training RNN-Transducer models.
   
Determine where cuda libraries are present. (On PARAM-Siddhi they were present in `/opt/cuda-11.0.2`, on MADHAVLAB1 they were present in `/usr/local/cuda`)
   
```bash
cd <espnet-root>/tools
conda activate espnet
cuda_root=/opt/cuda-11.0.2
. ./setup_cuda_env.sh $cuda_root
./installers/install_warp-transducer.sh
```



# 7. Check Installation
```bash
cd <espnet-root>/tools
python check_install.py
```      
