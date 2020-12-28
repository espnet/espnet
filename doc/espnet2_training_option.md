# Change the configuration for training
## Show usage
There are two ways to show the command line options: `--help` and `--print_config`

```bash
# Show the command line option
python -m espnet2.bin.asr_train --help
# Show the all configuration in yaml format
python -m espnet2.bin.asr_train --print_config
```

In this section, we use `espnet2.bin.asr_train` for an example, 
but the other training tools based on `Task` class have the same interface, 
so you can replace it to another command.

Note that ESPnet2 always selects`_` instead of `-` for the separation 
for the option name to avoid confusion.

```
# Bad
--batch-size
# Good
--batch_size
```

A notable feature of `--print_config` is that 
it shows the configuration parsing with the given arguments dynamically: 
You can look up the parameters for a **changeable** class.

```bash
% # Show parameters of Adam optimizer
% python -m espnet2.bin.asr_train --optim adam --print_config
...
optim: adam
optim_conf:
    lr: 0.001
    betas:
    - 0.9
    - 0.999
    eps: 1.0e-08
    weight_decay: 0
    amsgrad: false
...
% # Show parameters of ReduceLROnPlateau scheduler
% python -m espnet2.bin.asr_train --scheduler ReduceLROnPlateau --print_config
...
scheduler: reducelronplateau
scheduler_conf:
    mode: min
    factor: 0.1
    patience: 10
    verbose: false
    threshold: 0.0001
    threshold_mode: rel
    cooldown: 0
    min_lr: 0
    eps: 1.0e-08
...
```

## Configuration file
You can find the configuration files for DNN training in `conf/train_*.yaml`.

```bash
ls conf/
```

We adopt [ConfigArgParse](https://github.com/bw2/ConfigArgParse) for this configuration system. 
The configuration in YAML format has an equivalent effect to the command line argument. 
e.g. The following two are equivalent:

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

The state of the training process is saved at the end of every epoch as `checkpoint.pth` and 
the training process can be resumed from the start of the next epoch. 
Checkpoint includes the following states.

- Model state
- Optimizer states
- Scheduler states
- Reporter state
- torch.cuda.amp state (from torch=1.6)

## Transfer learning / Fine tuning using pretrained model

Use `--init_param <file_path>:<src_key>:<dst_key>:<exclude_keys>`

```bash
# Load all parameters
python -m espnet2.bin.asr_train --init_param model.pth
# Load only the parameters starting with "decoder"
python -m espnet2.bin.asr_train --init_param model.pth:decoder
# Load only the parameters starting with "decoder" and set it to model.decoder
python -m espnet2.bin.asr_train --init_param model.pth:decoder:decoder
# Set parameters to model.decoder
python -m espnet2.bin.asr_train --init_param decoder.pth::decoder
# Load all parameters excluding "decoder.embed"
python -m espnet2.bin.asr_train --init_param model.pth:::decoder.embed
# Load all parameters excluding "encoder" and "decoder.embed"
python -m espnet2.bin.asr_train --init_param model.pth:::encoder,decoder.embed
```

## Freeze parameters

```sh
python -m espnet2.bin.asr_train --freeze_param encoder.enc encoder.decoder
```

## Change logging interval
The result in the middle state of the training will be shown by the specified number:

```bash
python -m espnet2.bin.asr_train --log_interval 100
```

## Change the number of iterations in each epoch

By default, an `epoch` indicates using up whole data in the training corpus and 
the following steps will also run after training for every epoch:

- Validation
- Saving model and checkpoint
- Show result in the epoch

Sometimes the validation after training with a whole corpus is too coarse 
if using large corpus. 
For that case, `--num_iters_per_epoch` can restrict the number of iteration of each epoch.

```bash
python -m espnet2.bin.asr_train --num_iters_per_epoch 1000
```

Note that the training process can't be resumed at the middle of an epoch 
because data iterators are stateless, but don't worry it!
Our iterator is built at the start of each epoch 
and the random seed is fixed by the epoch number, just like:

```python
epoch_iter_factory = Task.build_epoch_iter_factory()
for epoch in range(max_epoch):
  iterator = epoch_iter_factory.build_iter(epoch)
```

Therefore, the training can be resumed at the start of the epoch.

## Weights & Biases integration

About Weights & Biases: https://docs.wandb.com/

1. Installation and setup

    See: https://docs.wandb.com/quickstart

    ```sh
    wandb login
    ```
1. Enable wandb

    ```sh
    python -m espnet2.bin.asr_train --use_wandb true
    ```

    and go to the shown URL.
1. [Option] To use HTTPS PROXY
    ```sh
    export HTTPS_PROXY=...your proxy
    export CURL_CA_BUNDLE=your.pem
    export CURL_CA_BUNDLE=   # Disable SSL certificate verification
    ```


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

The batch-size can be changed as follows:

```bash
# Change both of the batch_size for training and validation
python -m espnet2.bin.asr_train --batch_size 20
# Change the batch_size for validation
python -m espnet2.bin.asr_train --valid_batch_size 200
```

The behavior for batch-size during multi-GPU training is **different from that of ESPNet1**.

- ESPNet1: The batch-size will be multiplied by the number of GPUs.
  ```bash
  python -m espnet.bin.asr_train --batch_size 10 --ngpu 2 # Actual batch_size is 20 and each GPU devices are assigned to 10
  ```
- ESPnet2: The batch-size is not changed regardless of the number of GPUs.
  - Therefore, you should set a more number of batch-size than that of GPUs.
  ```bash
  python -m espnet.bin.asr_train --batch_size 10 --ngpu 2 # Actual batch_size is 10 and each GPU devices are assigned to 5
  ```

## Change mini-batch type
We adopt variable mini-batch size with considering the dimension of the input features
to make the best use of the GPU memory.

There are 5 types:

|batch_type|Option to change batch-size|Variable batch-size|Requirement|
|---|---|---|---|
|unsorted|--batch_size|No|-|
|sorted|--batch_size|No|Length information of features|
|folded|--batch_size|Yes|Length information of features|
|length|--batch_bins|Yes|Length information of features|
|numel|--batch_bins|Yes|Shape information of features|

Note that **--batch_size is ignored if --batch_type=length or --batch_type=numel**.

### `--batch_type unsorted`

This mode has nothing special feature and just creates constant-size mini-batches without any sorting by the length order. 
If you intend to use ESPnet as **not** Seq2Seq task, this type may be suitable.

Unlike the other mode, this mode doesn't require the information of the feature dimension.
In other words, it's not mandatory to prepare `shape_file`:

```bash
python -m espnet.bin.asr_train \
  --batch_size 10 --batch_type unsorted \
  --train_data_path_and_name_and_type "train.scp,feats,npy" \
  --valid_data_path_and_name_and_type "valid.scp,feats,npy" \
  --train_shape_file "train.scp" \
  --valid_shape_file "valid.scp"
```

This system might seem strange and you might also feel `--*_shape_file` is verbose
because the training corpus can be described totally only using `--*_data_path_and_name_and_type`.

From the viewpoint of the implementation, 
we separate the data source for the `Dataset` and `BatchSampler` in the term of PyTorch
and  `--*_data_path_and_name_and_type` and `--*_shape_file` correspond to them respectively.
From the viewpoint of the training strategy, 
because variable batch-size is supported according to the length/dimension of each feature,
thus we need to prepare the shape information before training.

### `--batch_type sorted`


This mode creates constant-size mini-batches with sorting by the length order. 
This mode requires the information of the length.


```bash
python -m espnet.bin.asr_train \
  --batch_size 10 --batch_type sorted \
  --train_data_path_and_name_and_type "train.scp,feats,npy" \
  --train_data_path_and_name_and_type "train2.scp,feats2,npy" \
  --valid_data_path_and_name_and_type "valid.scp,feats,npy" \
  --valid_data_path_and_name_and_type "valid2.scp,feats2,npy" \
  --train_shape_file "train_length.txt" \
  --valid_shape_file "valid_length.txt"
```

e.g. length.txt

```
sample_id1 1230
sample_id2 156
sample_id3 890
...
```

Where the fist column indicates the sample id and the second is the length of the corresponding feature.
You can see that `shape file` is input instead in our recipes. 

e.g. shape.txt

```
sample_id1 1230,80
sample_id2 156,80
sample_id3 890,80
...
```

This file describes the full information of the feature shape;
The first number is the length of the sequence and 
the second or later are the dimension of feature: `Length,Dim1,Dim2,...`.

Only the first number is referred for
`--batch_type sorted`, `--batch_type folded` and `--batch_type length`,
and the shape information is required only when `--batch_type numel`.


### `--batch_type folded`

**In ESPnet1, this mode is refered as seq.**


This mode creates mini-batch which has the size of `base_batch_size // max_i(1 + L_i // f_i)`. 
Where `L_i` is the maximum length in the mini-batch for `i`th feature and 
`f_i` is the `--fold length` corresponding to the feature. 
This mode requires the information of length.


```bash
python -m espnet.bin.asr_train \
  --batch_size 20 --batch_type folded \
  --train_data_path_and_name_and_type "train.scp,feats,npy" \
  --train_data_path_and_name_and_type "train2.scp,feats2,npy" \
  --valid_data_path_and_name_and_type "valid.scp,feats,npy" \
  --valid_data_path_and_name_and_type "valid2.scp,feats2,npy" \
  --train_shape_file "train_length.scp" \
  --train_shape_file "train_length2.scp" \
  --valid_shape_file "valid_length.scp" \
  --valid_shape_file "valid_length2.scp" \
  --fold_length 5000 \
  --fold_length 300
```

Note that the repeat number of `*_shape_file` must equal to the number of `--fold_length`, but 
**you don't need to input same number of shape files as the number of data file**. 
i.e. You can give it as follows:

```bash
python -m espnet.bin.asr_train \
  --batch_size 20 --batch_type folded \
  --train_data_path_and_name_and_type "train.scp,feats,npy" \
  --train_data_path_and_name_and_type "train2.scp,feats2,npy" \
  --valid_data_path_and_name_and_type "valid.scp,feats,npy" \
  --valid_data_path_and_name_and_type "valid2.scp,feats2,npy" \
  --train_shape_file "train_length.txt" \
  --valid_shape_file "valid_length.txt" \
  --fold_length 5000
```

In this example, the length of the first feature is considered while the second can be ignored. 
This technique can be also applied for `--batch_type length` and `--batch_type numel`.


### `--batch_type length`

**In ESPnet1, this mode is referred as frame.**


You need to specify `--batch_bins` to determine the mini-batch size instead of `--batch_size`. 
Each mini-batch has equal number of bins as possible counting by the total length in the mini-batch; 
i.e. `bins = sum(len(feat) for feats in batch for feat in feats)`. 
This mode requires the information of length.

```bash
python -m espnet.bin.asr_train \
  --batch_bins 10000 --batch_type length \
  --train_data_path_and_name_and_type "train.scp,feats,npy" \
  --train_data_path_and_name_and_type "train2.scp,feats2,npy" \
  --valid_data_path_and_name_and_type "valid.scp,feats,npy" \
  --valid_data_path_and_name_and_type "valid2.scp,feats2,npy" \
  --train_shape_file "train_length.txt" \
  --train_shape_file "train_length2.txt" \
  --valid_shape_file "valid_length.txt" \
  --valid_shape_file "valid_length2.txt" \
```


### `--batch_type numel`

**In ESPnet1, this mode is referred as bins.**

You need to specify `--batch_bins` to determine the mini-batch size instead of `--batch_size`. 
Each mini-batches has equal number of bins as possible 
counting by the total number of elements; 
i.e. `bins = sum(numel(feat) for feats in batch for feat in feats)`, 
where `numel` returns the infinite product of the shape of each feature; 
`shape[0] * shape[1] * ...`


```bash
python -m espnet.bin.asr_train \
  --batch_bins 200000 --batch_type numel \
  --train_data_path_and_name_and_type "train.scp,feats,npy" \
  --train_data_path_and_name_and_type "train2.scp,feats2,npy" \
  --valid_data_path_and_name_and_type  "valid.scp,feats,npy" \
  --valid_data_path_and_name_and_type  "valid2.scp,feats2,npy" \
  --train_shape_file "train_shape.txt" \
  --train_shape_file "train_shape2.txt" \
  --valid_shape_file "valid_shape.txt" \
  --valid_shape_file "valid_shape2.txt"
```

## Gradient accumulating
There are several ways to deal with larger model architectures than the capacity of your GPU device memory during training.

- Using a larger number of GPUs
- Using a half decision tensor
- Using [torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html)
- Gradient accumulating

Gradient accumulating is a technique to handle larger mini-batch than available size.

Split a mini-batch into several numbers and forward and backward for each piece and accumulate the gradients ony by one, 
while optimizer's updating is invoked every the number of forwarding just like following:

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

## Automatic Mixed Precision training

```bash
python -m espnet.bin.asr_train --use_amp true
```


## Reproducibility and determinization
There are some possibilities to make training non-reproducible.

- Initialization of parameters that come from PyTorch/ESPnet version difference.
- Reducing order for float values during multi GPUs training.
  - I don't know whether NCCL is deterministic or not.
- Random seed difference
  - We fixed the random seed for each epoch.
- CuDNN or some non-deterministic operations for CUDA: See https://pytorch.org/docs/stable/notes/randomness.html

By default, CuDNN performs deterministic mode in our training and it can be turned off by:

```bash
python -m espnet.bin.asr_train --cudnn_deterministic false
```
