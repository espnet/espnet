### Latest code:
The latest codebase is always under this Github branch. ```https://github.com/jctian98/espnet/tree/speechlm_main```  
Codebase: ```<espnet_root>/espnet2/speechlm```   
Recipe: ```<espnet_root>/egs2/librispeech/speechlm```   
Please Note: The whole ESPnet project is very large, but the codebase itself is totally self-consistent: it contains zero dependency to any other folders. The codebase folder itself is small and clean.

### Installation
(1) Ensure you have a valid Python environment. Here is an example of using Conda, but feel free to switch to other methods (e.g., docker with system Python).
```bash
cd <espnet_root>/tools
bash setup_anaconda.sh miniconda3 dev 3.11
source activate_python.sh
```

(2) Go to codebase folder:
```bash
cd <espnet_root>/espnet2/speechlm
```

(3) Install Pytorch. A newer version is appreciated.
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

(4) Install dependencies
```bash
pip install -r requirement.txt
```

(5) Install Flash attention. Recommend to build from source
```bash
# from pre-built wheel
pip install flash-attn --no-build-isolation 
# or from source
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

### Launch the training
(1) go to the directory.
```bash
cd <espnet_root>/egs2/librispeech/speechlm
```

(2) Remember to activate your Python environment  

(3) Prepare the dataset in the designed format.
```bash
bash prep.sh
```

(4) Collect the length of each example:
```bash
bash launch.sh --stage 1 --stop_stage 1
```

(5) Train the model
```bash
launch.sh --stage 2 --stop_stage 2
```

(6) Inference (ongoing)
```bash
launch.sh --stage 3 --stop_stage 3
```

### TODO list (as of Nov 17)
* Single-turn inference
* Classifier-Free Guidance in infernece
* Sequence Packing
* SFT data format support
* Support X-Codec
* Support more data loading interface
* Support TorchTitan-based Tensor/Pipeline Parallel (Low priority)
* Other efficiency optimization & large-scale test.
* Test on previous UALM TTA setup.
