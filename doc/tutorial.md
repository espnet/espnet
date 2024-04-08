## Common usages

### ESPnet1
Please first check [ESPnet1 tutorial](./espnet1_tutorial.md)

### ESPnet2
Please first check [ESPnet2 tutorial](./espnet2_tutorial.md)

### Multiple GPU TIPs
- Note that if you want to use multiple GPUs, the installation of [nccl](https://developer.nvidia.com/nccl) is required before setup.
- Currently, espnet1 only supports multiple GPU training within a single node. The distributed setup across multiple nodes is only supported in [espnet2](https://espnet.github.io/espnet/espnet2_distributed.html).
- We don't support multiple GPU inference. Instead, please split the recognition task for multiple jobs and distribute these split jobs to multiple GPUs.
- If you cannot get enough speed improvement with multiple GPUs, you should first check the GPU usage by `nvidia-smi`. If the GPU-Util percentage is low, the bottleneck will come from disk access. You can apply data prefetching by `--n-iter-processes 2` in your `run.sh` to mitigate the problem. Note that this data prefetching consumes a lot of CPU memory, so please be careful when you increase the number of processes.
- The behavior of batch size in ESPnet2 during multi-GPU training is different from that in ESPnet1. **In ESPnet2, the total batch size is not changed regardless of the number of GPUs.** Therefore, you need to manually increase the batch size if you increase the number of GPUs. Please refer to this [doc](https://espnet.github.io/espnet/espnet2_training_option.html#the-relation-between-mini-batch-size-and-number-of-gpus) for more information.

### Start from the middle stage or stop at the specified stage

`run.sh` has multiple stages, including data preparation, training, etc., so you may likely want to start
from the specified stage if some stages failed for some reason, for example.

You can start from the specified stage as follows and stop the process at the specified stage:

```bash
# Start from 3rd stage and stop at 5th stage
$ ./run.sh --stage 3 --stop-stage 5
```

### CTC, attention, and hybrid CTC/attention

ESPnet can easily switch the model's training/decoding mode from CTC, attention, and hybrid CTC/attention.

Each mode can be trained by specifying `mtlalpha` (espnet1) `ctc_weight` (espnet2):

- espnet1
```sh
# hybrid CTC/attention (default)
mtlalpha: 0.3

# CTC
mtlalpha: 1.0

# attention
mtlalpha: 0.0
```
- espnet2
```sh
# hybrid CTC/attention (default)
model_conf:
    ctc_weight: 0.3

# CTC
model_conf:
    ctc_weight: 1.0

# attention
model_conf:
    ctc_weight: 0.0
```

Decoding for each mode can be done using the following decoding configurations:

- espnet1
  ```sh
  # hybrid CTC/attention (default)
  ctc-weight: 0.3
  beam-size: 10

  # CTC
  ctc-weight: 1.0
  ## for best path decoding
  api: v1 # default setting (can be omitted)
  ## for prefix search decoding w/ beam search
  api: v2
  beam-size: 10

  # attention
  ctc-weight: 0.0
  beam-size: 10
  maxlenratio: 0.8
  minlenratio: 0.3
  ```

- espnet2
  ```sh
  # hybrid CTC/attention (default)
  ctc_weight: 0.3
  beam_size: 10

  # CTC
  ctc_weight: 1.0
  beam_size: 10

  # attention
  ctc_weight: 0.0
  beam_size: 10
  maxlenratio: 0.8
  minlenratio: 0.3
  ```

- The CTC mode does not compute the validation accuracy, and the optimum model is selected with its loss value, e.g.,
  - espnet1
    ```sh
    best_model_criterion:
    -   - valid
        - cer_ctc
        - min
    ```
  - espnet2
    ```sh
    ./run.sh --recog_model model.loss.best
    ```
- The pure attention mode requires setting the maximum and minimum hypothesis length (`--maxlenratio` and `--minlenratio`) appropriately. In general, if you have more insertion errors, you can decrease the `maxlenratio` value, while if you have more deletion errors, you can increase the `minlenratio` value. Note that the optimum values depend on the ratio of the input frame and output label lengths, which are changed for each language and each BPE unit.
- Negative `maxlenratio` can be used to set the constant maximum hypothesis length independently from the number of input frames. If `maxlenratio` is set to `-1`, the decoding will always stop after the first output, which can be used to emulate the utterance classification tasks. This is suitable for some spoken language understanding and speaker identification tasks.
- About the effectiveness of hybrid CTC/attention during training and recognition, see [2] and [3]. For example, hybrid CTC/attention is not sensitive to the above maximum and minimum hypothesis heuristics.
