# Distributed training
## Examples

### Single node with 4GPUs with distributed mode
```bash
% python -m espnet2.bin.asr_train --ngpu 4 --multiprocessing_distributed true
```

You can disable distributed mode and switch to threading based data parallel as following:

```bash
% python -m espnet2.bin.asr_train --ngpu 4 --multiprocessing_distributed false
```

### 2Nodes and 2GPUs for each node with multiprocessing distributed mode
Note that multiprocessing distributed mode assumes the same number of GPUs for each node.

```bash
(host1) % python -m espnet2.bin.asr_train \
    --ngpu 2 \
    --multiprocessing_distributed true \
    --dist_rank 0  \
    --dist_world_size 2  \
    --dist_master_addr host1  \
    --dist_master_port <any-free-port>
(host2) % python -m espnet2.bin.asr_train \
    --ngpu 2 \
    --dist_rank 1  \
    --multiprocessing_distributed true \
    --dist_world_size 2  \
    --dist_master_addr host1  \
    --dist_master_port <any-free-port>
```

#### About init method
See: https://pytorch.org/docs/stable/distributed.html#tcp-initialization

There are two ways to initialize:

- TCP initialization
   ```bash
   # These three are equivalent:
   --dist_master_addr <rank0-host> --dist_master_port <any-free-port>
   --dist_init_method "tcp://<rank0-host>:<any-free-port>"
   export MASTER_ADDR=<rank0-host> MASTER_PORT=<any-free-port>
   ```


- Shared file system initialization
   ```bash
   --dist_init_method "file:///nfs/some/where/filename"
   ```

   This initialization might be failed if the previous file is existing. I recommend you to use random file name to avoid to reuse it. e.g.

   ```bash
   --dist_init_method "file://$(pwd)/.dist_init_$(openssl rand -base64 12)"
   ```



### 2Nodes which have 2GPUs and 1GPU respectively
```bash
(host1) % python -m espnet2.bin.asr_train \
    --ngpu 1 \
    --multiprocessing_distributed false \
    --dist_rank 0  \
    --dist_world_size 3  \
    --dist_master_addr host1  \
    --dist_master_port <any-free-port>
(host1) % python -m espnet2.bin.asr_train \
    --ngpu 1 \
    --multiprocessing_distributed false \
    --dist_rank 1  \
    --dist_world_size 3  \
    --dist_master_addr host1  \
    --dist_master_port <any-free-port>
(host2) % python -m espnet2.bin.asr_train \
    --ngpu 1 \
    --multiprocessing_distributed false \
    --dist_rank 2  \
    --dist_world_size 3  \
    --dist_master_addr host1  \
    --dist_master_port <any-free-port>
```

### 2Nodes and 2GPUs for each node using `Slurm` with multiprocessing distributed

```bash
 % srun -c2 -N2 --gres gpu:2 \
    python -m espnet2.bin.asr_train --ngpu 2 --multiprocessing_distributed true \
    --dist_launcher slurm \
    --dist_init_method "file://$(pwd)/.dist_init_$(openssl rand -base64 12)"
```

### 5GPUs with 3nodes using `Slurm`
(Not tested)

```bash
% srun -n5 -N3 --gpus-per-task 1 \
    python -m espnet2.bin.asr_train --ngpu 1 --multiprocessing_distributed false  \
    --dist_launcher slurm \
    --dist_init_method "file://$(pwd)/.dist_init_$(openssl rand -base64 12)"
```

### 2Nodes and 2GPUs for each nodes using `MPI` with multiprocessing distributed

```bash
 % mpirun -np 2 -host host1,host2 \
    python -m espnet2.bin.asr_train --ngpu 2 --multiprocessing_distributed true \
    --dist_launcher mpi \
    --dist_init_method "file://$(pwd)/.dist_init_$(openssl rand -base64 12)"
```

## `espnet2.bin.launch`
Coming soon...

## Troubleshooting for NCCL with Ethernet case

-  `NCCL WARN Connect to 192.168.1.51<51890> failed : No route to host`
   - Reason: Firewall?
   - Need to free all ports?
- `NCCL INFO Call to connect returned Connection refused, retrying`
  - Reason: NIC is found, but connection is refused?
  - Set  `NCCL_SOCKET_IFNAME=<appropriate_interface>`
- `NCCL WARN Bootstrap : no socket interface found`
  - Reason: Any NIC are not found . (Maybe NCCL_SOCKET_IFNAME is incorrect)
  - Set `NCCL_SOCKET_IFNAME=<appropriate_interface>`.
- `NCCL WARN peer mapping resources exhausted`
  - ???
  - https://devtalk.nvidia.com/default/topic/970010/cuda-programming-and-performance/cuda-peer-resources-error-when-running-on-more-than-8-k80s-aws-p2-16xlarge-/post/4994583/#4994583


## The rules of `NCCL_SOCKET_IFNAME`
See: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html

- The default value is  `NCCL_SOCKET_IFNAME=^lo,docker`.
- Support two syntax: white list or black list
- White list e.g.: `NCCL_SOCKET_IFNAME=eth,em`
  - It's enough to specify the prefix only. You don't need to  set it as `eth0`.
- Black list e.g.: `^virbr,lo,docker`.
- If multiple network interfaces are found in your environment, the first is selected.
  - You can check your environment by `ifconfig` for example. https://www.cyberciti.biz/faq/linux-list-network-interfaces-names-command/
  - Note that `lo` is the first normally, so `lo` must be filtered.

My recommended setting for not virtual environment
-  `NCCL_SOCKET_IFNAME=en,eth,em,bond`
 -  Or, `NCCL_SOCKET_IFNAME=^lo,docker,virbr,vmnet,vboxnet,wl,ww,ppp`

|The prefix of network interface name|Note|
|---|---|
|lo|Loopback.|
|eth|Ethernet. Classically used.|
|em|Ethernet. Dell machine?|
|en|Ethernet (Used in recent linux. e.g CentOS7)|
|wlan|Wireless|
|wl|Wireless lan (Used in recent linux)|
|ww|Wireless wan (Used in recent linux)|
|ib|IP over IB|
|bond|Bonding of multiple ethernet |
|virbr|Virtual bridge|
|docker,vmnet,vboxnet|Virtual machine|
|ppp|Point to point|
