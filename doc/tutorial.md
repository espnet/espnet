## Outline

ESPnet is an end-to-end speech processing toolkit.
ESPnet uses [chainer](https://chainer.org/) as a main deep learning engine, 
and also follows [Kaldi](http://kaldi-asr.org/) style data processing, feature extraction/format, and recipes to provide a complete setup for speech recognition and other speech processing experiments.

## Installation

Install Kaldi, Python libraries and other required tools
```sh
$ cd tools
$ make -j
```

To use cuda (and cudnn), make sure to set paths in your `.bashrc` or `.bash_profile` appropriately.
```
CUDAROOT=/path/to/cuda

export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT
```

## Execution of example scripts
Move to an example directory under the `egs` directory.
We prepare several major ASR benchmarks including WSJ, CHiME-4, and TED.
The following directory is an example of performing ASR experiment with the VoxForge Italian Corpus.
```sh
$ cd egs/voxforge/asr1
```
Once move to the directory, then, execute the following main script:
```sh
$ ./run.sh
```
With this main script, you can perform a full procedure of ASR experiments including
- Data download
- Data preparation (Kaldi style, see http://kaldi-asr.org/doc/data_prep.html)
- Feature extraction (Kaldi style, see http://kaldi-asr.org/doc/feat.html)
- Dictionary and JSON format data preparation
- Training based on [chainer](https://chainer.org/).
- Recognition and scoring

### Use of GPU
If you use GPU in your experiment, set `--gpu` option in `run.sh` appropriately, e.g., 
```sh
$ ./run.sh --gpu 0
```
Default setup uses CPU (`--gpu -1`).

### Setup in your cluster
Change `cmd.sh` according to your cluster setup.
If you run experiments with your local machine, you don't have to change it.
For more information about `cmd.sh` see http://kaldi-asr.org/doc/queue.html.
It supports Grid Engine (`queue.pl`), SLURM (`slurm.pl`), etc.

## Installation using Docker

For GPU support nvidia-docker should be installed.

For Execution use the command 
```sh
$ cd egs/voxforge/asr1
$ ./run_in_docker.sh --gpu GPUID
```

If GPUID is set to -1, the program will run only CPU.

The file builds and loads the information into the Docker container. If any additional application is required, modify the Docker devel-file located at the tools folder.

To downgrade or use a private devel file, modify the name inside run_in_docker.sh

## References
Please cite the following articles.  
1. Suyoun Kim, Takaaki Hori, and Shinji Watanabe, "Joint CTC-attention based end-to-end speech recognition using multi-task learning," *Proc. ICASSP'17*, pp. 4835--4839 (2017)
2. Shinji Watanabe, Takaaki Hori, Suyoun Kim, John R. Hershey and Tomoki Hayashi, "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition," *IEEE Journal of Selected Topics in Signal Processing*, vol. 11, no. 8, pp. 1240-1253, Dec. 2017

