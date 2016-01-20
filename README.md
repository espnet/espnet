![Baidu Logo](/doc/baidu-research-logo-small.png)

[In Chinese 中文版](README.zh_cn.md)

# warp-ctc

A fast parallel implementation of CTC, on both CPU and GPU.

## Introduction

[Connectionist Temporal Classification](http://www.cs.toronto.edu/~graves/icml_2006.pdf)
is a loss function useful for performing supervised learning on sequence data,
without needing an alignment between input data and labels.  For example, CTC
can be used to train
[end-to-end](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf)
[systems](http://arxiv.org/pdf/1408.2873v2.pdf) for
[speech recognition](http://arxiv.org/abs/1512.02595),
which is how we have been using it at Baidu's Silicon Valley AI Lab.

![DSCTC](/doc/deep-speech-ctc-small.png)

The illustration above shows CTC computing the probability of an output
sequence "THE CAT ", as a sum over all possible alignments of input sequences
that could map to "THE CAT ", taking into account that labels may be duplicated
because they may stretch over several time steps of the input data (represented by
the spectrogram at the bottom of the image).
Computing the sum of all such probabilities explicitly would be prohibitively costly due to the
combinatorics involved, but CTC uses dynamic programming to dramatically
reduce the complexity of the computation. Because CTC is a differentiable function,
it can be used during standard SGD training of deep neural networks.

In our lab, we focus on scaling up recurrent neural networks, and CTC loss is an
important component. To make our system efficient, we parallelized the CTC
algorithm, as described in [this paper](http://arxiv.org/abs/1512.02595).
This project contains our high performance CPU and CUDA versions of the CTC loss,
along with bindings for [Torch](http://torch.ch/).
The library provides a simple C interface, so that it is easy to
integrate into deep learning frameworks.

This implementation has improved training scalability beyond the
performance improvement from a faster parallel CTC implementation. For
GPU-focused training pipelines, the ability to keep all data local to
GPU memory allows us to spend interconnect bandwidth on increased data
parallelism.

## Performance

Our CTC implementation is efficient compared with many of the other publicly available implementations.  It is
also written to be as numerically stable as possible.  The algorithm is numerically sensitive and we have observed
catastrophic underflow even in double precision with the standard calculation - the result of division of 
two numbers on the order of 1e-324 which should have been approximately one, instead become infinity 
when the denominator underflowed to 0.  Instead, by performing the calculation in log space, it is numerically
stable even in single precision floating point at the cost of significantly more expensive operations.  Instead of
one machine instruction, addition requires the evaluation of multiple transcendental functions.  Because of this,
the speed of CTC implementations can only be fairly compared if they are both performing the calculation the same
way.

We compare our performance with [Eesen](https://github.com/srvk/eesen/commit/68f2bc2d46a5513cce3c232a645292632a1b08f9), 
a CTC implementation built on 
[Theano](https://github.com/mohammadpz/CTC-Connectionist-Temporal-Classification/commit/904e8c72e15334887609d399254cf05a591d570f),
and a Cython CPU only implementation [Stanford-CTC](https://github.com/amaas/stanford-ctc/commit/c8859897336a349b6c561d2bf2d179fae90b4d67).
We benchmark the Theano implementation operating on 32-bit floating-point numbers and doing the calculation in log-space,
in order to match the other implementations we compare against.  Stanford-CTC was modified to perform the calculation
in log-space as it did not support it natively.  It also does not support minibatches larger than 1, so would require
an awkward memory layout to use in a real training pipeline, we assume linear increase in cost with minibatch size.

We show results on two problem sizes relevant to our English and Mandarin end-to-end models, respectively, where *T* represents the number of timesteps in the input to CTC, *L* represents the length of the labels for each example, and *A* represents the alphabet size.

On the GPU, our performance at a minibatch of 64 examples ranges from 7x faster to 155x faster than Eesen, and 46x to 68x faster than the Theano implementation.

### GPU Performance
Benchmarked on a single NVIDIA Titan X GPU.

| *T*=150, *L*=40, *A*=28           | warp-ctc  | Eesen   | Theano  |
|-----------------------------------|-------|---------|---------|
| *N*=1                             | 3.1 ms| .5 ms   | 67 ms |
| *N*=16                            | 3.2 ms| 6  ms   | 94 ms |
| *N*=32                            | 3.2 ms| 12 ms   | 119 ms |
| *N*=64                            | 3.3 ms| 24 ms   | 153 ms |
| *N*=128                           | 3.5 ms| 49 ms   | 231 ms |


| *T*=150, *L*=20, *A*=5000         | warp-ctc  | Eesen   | Theano  |
|-----------------------------------|-------|---------|---------|
| *N*=1                             | 7 ms  | 40   ms | 120 ms |
| *N*=16                            | 9 ms  | 619  ms | 385 ms |
| *N*=32                            | 11 ms | 1238 ms | 665 ms |
| *N*=64                            | 16 ms | 2475 ms | 1100 ms |
| *N*=128                           | 23 ms | 4950 ms | 2100 ms |

### CPU Performance

Benchmarked on a dual-socket machine with two Intel E5-2660 v3
processors - warp-ctc used 40 threads to maximally take advantage of the CPU resources.
Eesen doesn't provide a CPU implementation. We noticed that the Theano implementation was not
parallelizing computation across multiple threads.  Stanford-CTC provides no mechanism
for parallelization across threads.


| *T*=150, *L*=40, *A*=28           | warp-ctc  | Stanford-CTC   | Theano  |
|-----------------------------------|-------|---------|---------|
| *N*=1                             | 2.6 ms|  13 ms  | 15 ms |
| *N*=16                            | 3.4 ms|  208 ms | 180 ms |
| *N*=32                            | 3.9 ms|  416 ms | 375 ms |
| *N*=64                            | 6.6 ms|  832 ms | 700 ms |
| *N*=128                           |12.2 ms| 1684 ms | 1340 ms |


| *T*=150, *L*=20, *A*=5000         | warp-ctc  | Stanford-CTC   | Theano  |
|-----------------------------------|-------|---------|---------|
| *N*=1                             | 21 ms |  31 ms  | 850 ms  |
| *N*=16                            | 37 ms |  496 ms | 10800 ms|
| *N*=32                            | 54 ms |  992 ms | 22000 ms|
| *N*=64                            | 101 ms| 1984 ms | 42000 ms|
| *N*=128                           | 184 ms| 3968 ms | 86000 ms|





## Interface

The interface is in [`include/ctc.h`](include/ctc.h).
It supports CPU or GPU execution, and you can specify OpenMP parallelism
if running on the CPU, or the CUDA stream if running on the GPU. We
took care to ensure that the library does not perform memory
allocation internally, in order to avoid synchronizations and
overheads caused by memory allocation.

## Compilation

warp-ctc has been tested on Ubuntu 14.04 and OSX 10.10.  Windows is not supported
at this time.

First get the code:

```
git clone https://github.com/baidu-research/warp-ctc.git
cd warp-ctc
```

create a build directory:

```
mkdir build
cd build
```

if you have a non standard CUDA install `export CUDA_BIN_PATH=/path_to_cuda` so that CMake detects CUDA and
to ensure Torch is detected, make sure `th` is in `$PATH`

run cmake and build:

```
cmake ../
make
```

The C library and torch shared libraries should now be built along with test
executables.  If CUDA was detected, then `test_gpu` will be built; `test_cpu`
will always be built.

## Tests

To run the tests, make sure the CUDA libraries are in `LD_LIBRARY_PATH` (`DYLD_LIBRARY_PATH` for OSX).

The Torch tests must be run from the `torch_binding/tests/` directory.

## Torch Installation

```luarocks make torch_binding/rocks/warp-ctc-scm-1.rockspec```

You can also install without cloning the repository using

```luarocks install http://raw.githubusercontent.com/baidu-research/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec```

There is a Torch CTC [tutorial](torch_binding/TUTORIAL.md).

## Contributing

We welcome improvements from the community, please feel free to submit pull
requests.

## Known Issues  / Limitations

The CUDA implementation requires a device of at least compute capability 3.0.

The CUDA implementation supports a maximum label length of 639 (timesteps are
unlimited).
