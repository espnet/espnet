# Change the configuration for training
## Show usage
There are two ways to know the command line options: `--help` and `--print_config`

```bash
# Show the command line option
python -m espnet2.bin.asr_train --help
# Show the all configuration in yaml format
python -m espnet2.bin.asr_train --print_config
```

In this section, we use `espnet2.bin.asr_train` for an example, but the other training tools based on `Task` class has the same interface.

Note that ESPnet2 always selects`_` instead of `-` for the separation for the name of an option to avoid confusion.

A notable feature of `--print_config` is that it shows the configuration parsing with the given arguments dynamically: You can look up the parameters for a changeable class for example.

```bash
# Show parameters of Adam optimizer
python -m espnet2.bin.asr_train --optim adam --print_config
# Show parameters of ReduceLROnPlateau scheduler
python -m espnet2.bin.asr_train --scheduler ReduceLROnPlateau --print_config
```

## Configuration file
You can find the configuration files for DNN training in `conf/train_*.yaml`.

```bash
ls conf/
```

We adopt [ConfigArgParse](https://github.com/bw2/ConfigArgParse) for this configuration system. The configuration in YAML format has an equivalent effect to the command line argument. e.g. The following two are equivalent:

```yaml
# config.yaml
foo: 3
bar: 4
```

```bash
python -m espnet2.bin.asr_train --config conf/config.yaml
python -m espnet2.bin.asr_train --foo 3 --bar 4
```


## Change the configuration for dict type value

Some parameters are named as `*_conf`, e.g. `optim_conf`, `decoder_conf` and they has the `dict` type value. We also provide a way to configure the nested value in such a dict object.

```bash
# e.g. Change parameters one by one
python -m espnet2.bin.asr_train --optim_conf lr=0.1 --optim_conf rho=0.8
# e.g. Give the parameters in yaml format
python -m espnet2.bin.asr_train --optim_conf "{lr: 0.1, rho: 0.8}"
```

## Resume training process

```bash
python -m espnet2.bin.asr_train --resume true
```

The state of the training process is saved at the end of every epoch as `checkpoint.pth` and the training process can be resumed from the start of the next epoch. Checkpoint includes the following states.

- Model state
- Optimizer states
- Scheduler states
- Reporter state
- apex.amp state

## Change logging interval
The result in the middle state of the training will be shown by the specified number:

```bash
python -m espnet2.bin.asr_train --log_interval 100
```

## Change the number of iterations in each epoch

By default, an ``epoch`` indicates using up whole data in the training corpus and the following steps will also run after training for every epoch:

- Validation
- Saving model and checkpoint
- Show result in the epoch

Sometimes the examination after training with a whole corpus is too coarse if using large corpus: `--num_iters_per_epoch` restrict the number of iteration of each epoch.

```bash
python -m espnet2.bin.asr_train --num_iters_per_epoch 1000
```

Note that the training process can't be resumed at the middle of an epoch because data iterators are stateless, but instead of it, the iterators for each epoch can be built with the specific epoch number deterministically, just like:

```python
epoch_iter_factory = Task.build_epoch_iter_factory()
for epoch in range(max_epoch):
  iterator = epoch_iter_factory.build_iter(epoch)
```

Therefore, the training can be resumed at the start of the epoch.

## Multi GPUs

```bash
python -m espnet2.bin.asr_train --ngpu 2
```

Just using `CUDA_VISIBLE_DEVICES` to specify the device number:

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m espnet2.bin.asr_train --ngpu 2
```

About distributed training, see [Distributed training](espnet2_distributed.md).

## The relation between mini-batch size and number of GPUs

In ESPnet1, we support three types of mini-batch type:

- --batch-count seq
- --batch-count bin
- --batch-count frame

For now, ESPnet2 supports only `seq` mode. The batch-size can be changed as follows:

```bash
# Change both of the batch_size for training and validation
python -m espnet2.bin.asr_train --batch_size 20
# Change the batch_size for validation
python -m espnet2.bin.asr_train --valid_batch_size 200
```

The behavior for batch-size when multi-GPU mode is **different from that of ESPNe1**.

- ESPNet1: The batch-size will be multiplied by the number of GPUs.
  ```bash
  python -m espnet.bin.asr_train --batch_size 10 --ngpu 2 # Actual batch_size is 20 and each GPU devices are assigned to 10
  ```
- ESPnet2: The batch-size is not changed regardless of the number of GPUs.
  - Therefore, you should more number of batch-size than the number of GPUs.
  ```bash
  python -m espnet.bin.asr_train --batch_size 10 --ngpu 2 # Actual batch_size is 10 and each GPU devices are assigned to 5
  ```

Note that even espnet1, if using `bin` or `frame` batch-count, this changing of batch_size is not done.

## Gradient accumulating
There are several ways to deal with larger model architectures than the capacity of your GPU device memory during training.

- Using a larger number of GPUs
- Using a half decision tensor
- Using [torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html)
- Gradient accumulating

Gradient accumulating is a technique to handle larger mini-batch than available size.

Split a mini-batch into several numbers and forward and backward for each pieces and accumulate the gradients, while optimizer's updating is invoked every the number of forwarding just like following:

```python
# accum_grad is the number of pieces
for i, batch in enumerate(iterator):
  loss = net(batch)
  (loss / accum_grad).backward() # Gradients are accumulated
  if i % accum_grad:
    optim.update()
    optim.zero_grads()
```

Give `--accum_grad <int>` to use this option.

```bash
python -m espnet.bin.asr_train --accum_grad 2
```

 The effective batch_size becomes **almost** same as `accum_grad * batch_size` except for:

- The random state
- Some statistical layers based on mini-batch e.g. BatchNormalization
- The case that the batch_size is not unified for each iteration.

## Using apex: mixed-precision training
See also: https://github.com/NVIDIA/apex

```bash
python -m espnet.bin.asr_train --train_dtype float16 # Just training with half precision
python -m espnet.bin.asr_train --train_dtype float32 # default
python -m espnet.bin.asr_train --train_dtype float64
python -m espnet.bin.asr_train --train_dtype O0 # opt_level of apex
python -m espnet.bin.asr_train --train_dtype O1 # opt_level of apex
python -m espnet.bin.asr_train --train_dtype O2 # opt_level of apex
python -m espnet.bin.asr_train --train_dtype O3 # opt_level of apex
```

## Reproducibility and determinization
There are some possibilities to make training non-reproducible.

- Initialization of parameters that come from PyTorch/ESPnet version difference.
- Reducing order for float values when multi GPUs
  - I don't know whether NCCL is deterministic or not.
- Random seed difference
  - We fixed the random seed for each epoch, so the randomness should be reproduced even if the process is resumed.
- CuDNN or some non-deterministic operations for CUDA: See https://pytorch.org/docs/stable/notes/randomness.html

By default, CuDNN performs non-deterministic mode and it can be changed by:

```bash
python -m espnet.bin.asr_train --cudnn_deterministic true
```
