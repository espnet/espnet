## Use adapters for ASR in ESPnet2

In this tutorial, we will introduce several options to use adapters for Automatic Speech Recognition (ASR) in ESPnet. Available options are:
- Insert adapters to specific layers with a pre-trained s3prl frontend for adaptation of downstream tasks*.
- Enable automatic adapter insertion via `FindAdaptNet`.

Note this is done for ASR training, so at __stage 11__ of ESPnet2 recipes.

### 0. Why use adapters?

Adapters are small bottleneck layers that can be inserted into any layer of a neural network. They are trained to adapt the network to a specific downstream task. For more details, please refer to the [adapter paper](https://arxiv.org/abs/1902.00751). Used with large acoustic frontend, they can be used to adapt the frontend to a specific downstream task in a __parameter-efficient manner__. In general, the benefits of using adapters are:
- Learn robust recognition models with pre-trained weights and limited computational resources.
- Adapt to various downstream tasks efficiently
- Adapt to low-resource languages easily.
- ...

### 1. Use a s3prl frontend with adapters

__Step 1__: specify in your config yaml file (they are usually placed in `espnet/egs2/some_dataset/some_model/conf/`) that
```

add_adapters: True
adapter_down_dim: some_dim
adapter_layer: [l0, l1 ... ln]
```
where `some_dim` is the dimension of the adapter bottleneck layer, and `adapter_layer` is a list of layers to insert adapters. For example, if you want to insert adapters to the 1st, 3rd, and 5th layers of the frontend, you can set `adapter_layer: [1, 3, 5]` (*0-indexed*).


__Step 2__: run __stage 11__ of your recipe, with `./asr.sh --stage 11 ...`

### 2. Use FindAdaptNet  for automatic adapter insertion

__Step 1__: specify in your config yaml file (they are usually placed in `espnet/egs2/some_dataset/some_model/conf/`) that
```
    adapter_find: True
    adapter_num: some_num
```

where `some_num` is the number of adapters to insert. Do not have `add_adapters: True` in your config file.


__Step 2__: run __stage 11__ of your recipe, with `./asr.sh --stop-stage 11 ...`, specify how many epochs you would like FIndAdaptNet to train to find best insertion configuration. *Empirically, we find that 10 epochs is enough to find a converged configuration*.

>FindAdaptNet would generate a new yaml file with the best adapter insertion whose name would have the original file name as prefix + `_adapt.yaml`. So if your file was originally named `asrconfig.yaml`, the generated config file would be named as `asrconfig_adapt.yaml`.

__Step 3__: Run __stage 11__ again with `./asr.sh --stage 11 ...`, make sure to specify the **generated config file** as the asr config file in this run.


### 3. Examples
As shown in #1 and #2, using adapters is really easy. An example yaml config file could be found at `espnet/egs2/mini_an4/asr1/conf/adapter_example.yaml` and the shell file to run the config is at `espnet/egs2/mini_an4/asr1/run_adapter.sh`.
