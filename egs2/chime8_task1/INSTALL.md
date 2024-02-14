### <a id="espnet_installation"> Installation  </a>

Firstly, clone ESPNet. <br/>
```bash
git clone https://github.com/espnet/espnet/
```
Next, ESPNet must be installed, go to espnet/tools. <br />
This will install a new environment called espnet with Python 3.9.2
```bash
cd espnet/tools
./setup_anaconda.sh venv "" 3.9.2
```
Activate this new environment.
```bash
source ./venv/bin/activate
```
Then install ESPNet with Pytorch 1.13.1 be sure to put the correct version for **cudatoolkit**.
```bash
make TH_VERSION=1.13.1 CUDA_VERSION=11.6
```
If you plan to train the ASR model, you would need to compile Kaldi. Otherwise you can
skip this step. Go to the `kaldi` directory and follow instructions in `INSTALL`.
```bash
cd kaldi
cat INSTALL
```
Finally, get in this recipe **asr1 folder** and install other baseline required packages (e.g. lhotse) using this script:
```bash
cd ../egs2/chime8_task1/asr1
./local/install_dependencies.sh
```
You should be good to go !

⚠️ if you encounter any problem have a look at [HELP.md](HELP.md) here. <br>
Or reach us, see [README.md](./README.md).</a>
