![Baidu Logo](/doc/baidu-research-logo-small.png)

[In English](README.md)

# warp-ctc

Warp-CTC是一个可以应用在CPU和GPU上高效并行的CTC代码库 （library）
介绍
CTC[Connectionist Temporal Classification](http://www.cs.toronto.edu/~graves/icml_2006.pdf)作为一个损失函数，用于在序列数据上进行监督式学习，不需要对齐输入数据及标签。比如，CTC可以被用来训练端对端的语音识别系统，这正是我们在百度硅谷试验室所使用的方法。
[端到端](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf)
[系统](http://arxiv.org/pdf/1408.2873v2.pdf)
[语音识别](http://arxiv.org/abs/1512.02595)

![DSCTC](/doc/deep-speech-ctc-small.png)

上图展示了CTC计算输出序列（“THE CAT”）概率的过程，是对可能映射成“THE CAT”的所有可能输入序列对齐的和。这一过程考虑了标签会被复制的可能性，因为标签有可能在输入数据的几个时间步（time steps)时被拉伸 （请见上图底部的声谱图）。由于涉及到了组合学，计算所有可能概率的和的成本会很高，但是CTC运用了动态规划以大幅降低计算的复杂性。作为一个可微函数，CTC可以被用于深度神经网络的标准SGD训练。
我们实验室专注于递归神经网络（RNN）的可扩展性 （scalibility), 而CTC损失函数是其中很重要的一部分。为了让我们的系统更有效率，我们并行处理了CTC算法，正如这篇文章中所描述的 。这个项目包含了我们的高性能CPU以及CUDA版本的CTC损失函数, 以及绑定的Torch. 该代码库提供了简单的C接口，易于与深度学习框架整合。

这种执行方式提高了训练的的可扩展性，超过了并行CTC的实现方式。对于以GPU为核心的训练， 我们可用所有的的网络带宽来增加数据的可并行性。
性能
相起其他的开源工具，Warp-CTC的实现方式相对高效，且代码的数值稳定性也较好。因为CTC本身对数值较为敏感，因此即使使用双精度标准计算，也会出现下溢 (underflow)的情况。 具体来说，两个数值趋近于无穷小且相近的数字相除的结果应该大约为1，却因为分母接近为0而变成无穷。 然而，如果直接取对数执行运算，CTC会在数值上较为稳定，虽然会在单精度浮点中以高成本运算为代价。
我们将Warp-CTC和[Eesen](https://github.com/srvk/eesen/commit/68f2bc2d46a5513cce3c232a645292632a1b08f9) (建立在[Theano](https://github.com/mohammadpz/CTC-Connectionist-Temporal-Classification/commit/904e8c72e15334887609d399254cf05a591d570f)上的CTC)以及仅运行[Stanford-CTC](https://github.com/amaas/stanford-ctc/commit/c8859897336a349b6c561d2bf2d179fae90b4d67)的Cython CPU进行了比较。为了进行比较，我们对在32位浮点数上运行的Theano进行了基准测试，并且取对数计算。 而Stanford-CTC由于本身不支持对数运算，因此需要被修改。而且它也不支持大于1的迷你批处理 （minibatches), 所以需要在真正的训练流水线上布局非常规内存（我们假设成本与迷你批处理的规模是成正线性关系）。
我们在Deep Speech 2中分别展示了英文及中文端对端模型的结果, 其中T代表输入CTC的时间步数量，L代表每个例子的标签长度，A代表字母数量。
在GPU上，Warp-CTC对64个例子迷你批处理的表现比Eesen快7倍到155倍，比Theano快46倍到68倍
### GPU性能
单核NVIDIA Titan X GPU基准测试

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

### CPU性能
在一台有两个Intel E5-2660 v3处理器的双槽机上进行基准测试。Warp-CTC用了40个线程从而最大化了对CPU资源的利用。Eesen没有提供CPU实现方式。我们注意到Theano没有在多线程上进行并行计算。同样，Stanford－CTC没有提供多线程并行计算的机制。 

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

## 接口
接口在[`include/ctc.h`](include/ctc.h)中，它支持在CPU或者GPU上执行。 如果是在CPU上运行，可以指定OpenMP并行计算; 如果是在GPU上运行，请用CUDA stream。 为避免内存分配而导致的竞争及间接成本，我们会确保代码库不会在内部进行内存分配。 
## 编译器
Warp-CTC已经在Ubuntu 14.04以及OSX 10.10进行了测试，现不支持Windows. 
首先，请获取代码

```
git clone https://github.com/baidu-research/warp-ctc.git
cd warp-ctc
```

创建目录

```
mkdir build
cd build
```

假如使用非标准CUDA，请安装 `export CUDA_BIN_PATH=/path_to_cuda` 以便被CMake检测。且确保Torch被监测到，注意（`th` is in `$PATH`）
运行cmake, 创建

```
cmake ../
make
```

现在，C代码库以及与torch分享的代码库应当和测试可执行文件一同被创建。假如CUDA被检测到，test_gpu则被创建。
测试
为了运行测试，确保CUDA代码库在`LD_LIBRARY_PATH` (`DYLD_LIBRARY_PATH` for OSX)中。
Torch测试必须在 `torch_binding/tests/` 目录中运行。
## Torch安装

```luarocks make torch_binding/rocks/warp-ctc-scm-1.rockspec```

即使不复制存储库（repository)，你也可以安装

```luarocks install http://raw.githubusercontent.com/baidu-research/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec```

[请见Torch CTC教程](torch_binding/TUTORIAL.zh_cn.md)。

## 限制
CUDA的执行需要至少3.0的计算能力， 所支持的标签长度最大值为639 （时间步数是有限的）。

最后我们欢迎大家提出宝贵的意见及建议以改进我们的开源服务。

在此鸣谢新智元编译 [http://chuansong.me/account/AI_era](http://chuansong.me/account/AI_era)允许我们参考部分译文，[http://chuansong.me/n/2168385](http://chuansong.me/n/2168385)
