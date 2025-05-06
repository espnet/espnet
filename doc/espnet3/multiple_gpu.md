## ESPnet3: Multi-GPU and Multi-Node Training

ESPnet2 originally supported multi-node, multi-GPU training using shell-based job scripts. In contrast, ESPnet3 achieves the same functionality **without relying on shell scripts**, using built-in capabilities from **PyTorch Lightning**.

PyTorch Lightning natively supports both **multi-GPU** and **multi-node** training. Users can simply configure their trainer in the YAML file to activate distributed training.

### ✅ Example: Multi-GPU, Multi-Node Trainer Configuration

```yaml
trainer:
  accelerator: gpu
  devices: 4            # Number of GPUs per node
  num_nodes: 2          # Number of nodes (machines)
  strategy: ddp         # Use distributed data parallel
  logger:
    - _target_: lightning.pytorch.loggers.TensorBoardLogger
      save_dir: ${expdir}/tensorboard
      name: tb_logger
```

### ⚠️ Important Notes for Cluster Environments

If you're using a cluster environment such as SLURM, make sure to:
- Request the correct number of GPUs and nodes in your SLURM submission script
- Use `srun` to launch the training script (so PyTorch Lightning can manage worker communication)

Example SLURM command:
```bash
srun python train.py
```

---

### ✅ Multi-GPU Inference with ESPnet3 Parallel API

For inference or evaluation with multiple GPUs, you can also use the `espnet3.parallel` module to parallelize the job across GPUs.

```python
from dask.distributed import get_worker, WorkerPlugin, as_completed
from espnet3.parallel import (
    set_parallel, get_client, parallel_map
)

# Set parallel configuration
set_parallel(config.parallel)

# Initialize large instance for each worker
class LargeInstancePlugin(WorkerPlugin):
    def setup(self, worker):
        worker.model = load_model()
        worker.model.to("cuda")
        worker.dataset = load_dataset()

def inference(idx):
    data = get_worker().dataset
    model = get_worker().model
    out_text = model(data['audio']['array'])[0][0]
    return {
        "hyp": out_text,
        "ref": data['text']
    }

# Process
with get_client(LargeInstancePlugin()) as client:
    results = parallel_map(inference, list(range(len(test_data))))

hyps = [r['hyp'] for r in results]
refs = [r['ref'] for r in results]

wer = compute_wer(refs, hyps)
print(wer)
```

```yaml
parallel_gpu:
  env: slurm
  n_workers: 4
  options:
    queue: gpu
    cores: 8
    processes: 1
    memory: 16GB
    walltime: 01:00:00
    job_extra_directives:
      - "--gres=gpu:1"
```

This enables scalable evaluation or decoding using the same interface as training, without additional script modifications.

---

By leveraging PyTorch Lightning and the ESPnet3 parallel API, you can easily scale your training and inference workloads across multiple GPUs and nodes—without needing to write custom shell scripts.
