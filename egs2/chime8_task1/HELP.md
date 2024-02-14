## <a id="common_issues"> Common Issues </a>

⚠️ If you use `run.pl` please check GSS logs when it is running and ensure you don't have any other processes on the GPUs.

1. `AssertionError: Torch not compiled with CUDA enabled` <br> for some reason you installed Pytorch without CUDA support. <br>
 Please install Pytorch with CUDA support as explained in [pytorch website](https://pytorch.org/).
2. `ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'YOUR_PATH/espnet/tools/venv/lib/pyth
on3.9/site-packages/numpy-1.23.5.dist-info/METADATA'`. This is due to numpy installation getting corrupted for some reason.
You can remove the site-packages/numpy- folder manually and try to reinstall numpy 1.23.5 with pip.
3. `FileNotFoundError: [Errno 2] No such file or directory: 'PATH2YOURESPNET/espnet/tools/venv/bin/sox'
` during CHiME-6 generation from CHiME-5, `correct_signals_for_clock_drift.py` script: try to install conda sox again, via `conda install -c conda-forge sox`.
4. `ModuleNotFoundError: No module named 's3prl'` for some reason s3prl did not install, run `YOUR_ESPNET_ROOT/tools/installers/install_s3prl.sh`
5. `Command 'gss' not found` for some reason gss did not install, you can run `YOUR_ESPNET_ROOT/tools/installers/install_gss.sh`
7. `wav-reverberate command not found` you need to install Kaldi. go to `YOUR_ESPNET_ROOT/tools/kaldi` and follow the instructions
in `INSTALL`.
8. `WARNING  [enhancer.py:245] Out of memory error while processing the batch` you got out-of-memory (OOM) when running GSS.
You could try changing parameters as `gss_max_batch_dur` and in local/run_gss.sh `context-duration`
(this latter could degrade results however). See `local/run_gss.sh` for more info. Also it could be that your GPUs are set in shared mode
and all jobs are placed in the same GPU. You need to set them in exclusive mode.
9. **Much worse WER than baseline** and you are using `run.pl`. **Check the GSS results and logs it is likely that enhancement failed**.
**GSS currently does not work well if you use multi-gpu inference and your GPUs are in shared mode**. You need to run `set nvidia-smi -c 3` for each GPU.

## Number of Utterances for Each Dataset in this Recipe
### Training Set
#### all
- kaldi/train_all_ihm: 175403
- kaldi/train_all_ihm_rvb: 701612
- kaldi/train_all_mdm_ihm: 2150180
- kaldi/train_all_mdm_ihm_rvb: 2851792
- kaldi/train_all_mdm_ihm_rvb_gss: 2914483 (used for training here)
#### chime6

- kaldi/chime6/train/mdm: 1403340
- kaldi/chime6/train/gss: 62691
- kaldi/chime6/train/ihm: 118234
#### mixer6

- kaldi/mixer6/train/mdm: 571437
- kaldi/mixer6/train/ihm: 57169

### Development Set
#### all
- kaldi/dev_ihm_all; kaldi/dev_all_gss: 25121
#### chime6
- kaldi/chime6/dev/gss (used for validation here); kaldi/chime6/dev/ihm: 6644
#### dipco
- kaldi/dipco/dev/gss; kaldi/dipco/dev/ihm: 3673
#### mixer6
- kaldi/mixer6/dev/gss; kaldi/mixer6/dev/ihm: 14804

## Memory Consumption (Useful for SLURM etc.)

Figures kindly reported by Christoph Boeddeker, running this baseline code
on Paderborn Center for Parallel Computing cluster (which uses SLURM).
These figures could be useful to anyone that uses job schedulers and clusters
for which resources are assigned strictly (e.g. job killed if it exceed requested
memory resources).

Used as default:
 - train: 3G mem
 - cuda: 4G mem (1 GPU)
 - decode: 4G mem

scripts/audio/format_wav_scp.sh:
 - Some spikes to the range of 15 to 17 GB

`${python} -m espnet2.bin.${asr_task}_inference${inference_bin_tag`}:
 - Few spikes to the 9 to 11 GB range.


## Using your own Speech Separation Front-End with the pre-trained ASR model.

Some suggestions from Naoyuki Kamo see https://github.com/espnet/espnet/pull/4999 <br>
There are two possible approaches.

1. After obtaining the output of the baseline GSS enhancement.
   1. (you are more familiar with Kaldi): copy data/kaldi/{chime6,dipco,mixer6}/{dev,eval}/gss using utils/copy_data_dir.sh and then substitute the
   file paths in each `wav.scp` manifest file with the ones produced by your approach.
   2. (you are more familiar with lhotse): copy data/lhotse/{chime6,dipco,mixer6}/{dev,eval}/gss lhotse manifests and then
   replace the recordings manifests with the paths to your own recordings.
2. Directly create your Kaldi or lhotse manifests for the ASR decoding, you can follow
either the "style" of the baseline GSS ones or the ones belonging to close-talk mics.

To evaluate the new enhanced data, e.g. `kaldi/chime6/dev/my_enhanced`, you need to include it into `asr_tt_set` in `run.sh` or
from command line: `run.sh --stage 3 --asr-tt-set "kaldi/chime6/dev/gss" --decode-train dev --use-pretrained popcornell/chime7_task1_asr1_baseline --asr-dprep-stage 4`.
