## ESPnet3: Dynamic Batching Guide

This document explains how to use dynamic batch sizing in espnet3.
It covers the following sections:

1. Why dynamic batching is necessary for speech
2. How ESPnet3 implements dynamic batching using Lhotse

---

### 1. Why Dynamic Batching is Needed

Unlike text or image data, speech data can vary greatly in length. This leads to significant differences in the size of acoustic feature matrices. For example, consider two utterances:
- One 30-second long
- One 5-second long
When batching together, the 5-second audio needs to be padded to match the 30-second audio in the batch. 

![](./images/dynamic_batch_sec.png)

If the average length of utterances is 6 seconds, and batch size is set to 5, the total number of frames in a batch would usually be about 3000. But if a batch contains a single 30-second example, it would have 3000 frames in just one sample, which could lead to GPU Out-Of-Memory (OOM) errors.

To address this, **dynamic batching** controls the total number of frames per batch, adjusting the number of samples accordingly:
- 1 sample if it's a long 30-second audio
- 5 samples if each is 6 seconds long

This approach maximizes GPU memory utilization without risking OOM.

---

### 2. How ESPnet3 Implements Dynamic Batching

To implement dynamic batching, each audio segment must have metadata indicating its duration. ESPnet2 collected this information in a "collect-stats" stage, but ESPnet3 removes this step to simplify design and avoid tightly coupled dependencies between components.

Instead, ESPnet3 adopts **Lhotse**, which stores audio duration metadata in CutSet manifests.

For example, using Lhotse's `DynamicBucketingSampler`, ESPnet3 can efficiently sample batches where the **total duration is fixed**, but the number of samples per batch varies.

In ESPnet3, all data loading logic, including sampler and collate function, is defined through configuration files such as `egs3/config.yaml`. This allows users to enable dynamic batching without modifying training scripts.

---

### ✅ Sample Config: Dataloader with Dynamic Bucketing

```yaml
dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1

  train:
    dataset:
      _target_: espnet3.data.dataset.ESPnet3Dataset
      path: /path/to/train_dataset
    sampler:
      _target_: lhotse.dataset.sampling.DynamicBucketingSampler
      max_duration: 30.0
      drop_last: true
      shuffle: true
    batch_size: null  # required when using dynamic sampler
    num_workers: 4

  valid:
    dataset:
      _target_: espnet3.data.dataset.ESPnet3Dataset
      path: /path/to/valid_dataset
    sampler:
      _target_: lhotse.dataset.sampling.SimpleCutSampler
      max_cuts: 20
      shuffle: false
    batch_size: null
    num_workers: 4
```

---

### ✅ Integration with Training Pipeline

The model class (`LitESPnetModel`) automatically receives dataloaders created based on this configuration. This is handled internally in `ESPnet3/trainer/model.py`, so the user does not need to manually construct samplers in their training scripts.

By leveraging Lhotse and Hydra-style configuration, ESPnet3 enables dynamic batching with clean abstraction, removing the need for a separate collect-stats stage and supporting scalable and memory-efficient training workflows.

