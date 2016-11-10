
# TensorFlow binding for WarpCTC

This package provides TensorFlow kernels that wrap the WarpCTC
library.  Kernels are provided for both the CTCLoss op already in
TensorFlow, as well as a new WarpCTC op provided in this package.  The
WarpCTC op has an interface that more closely matches the native
WarpCTC interface than TensorFlow's CTCLoss op. Note that the CTCLoss
op expects the reserved blank label to be the largest value while the
WarpCTC op takes the reserved blank label value as an attribute which
defaults to `0`.

## Installation

To build the kernels it is necessary to have the TensorFlow source
code available, since TensorFlow doesn't currently install the
necessary headers to handle the SparseTensor that the CTCLoss op uses
to input the labels.  You can retrieve the TensorFlow source from
github.com:

```bash
git clone https://github.com/tensorflow/tensorflow.git
```

Tell the build scripts where you have the TensorFlow source tree by
setting the `TENSORFLOW_SRC_PATH` environment variable:

```bash
export TENSORFLOW_SRC_PATH=/path/to/tensorflow
```

`WARP_CTC_PATH` should be set to the location of a built WarpCTC
(i.e. `libwarpctc.so`).  This defaults to `../build`, so from within a
new warp-ctc clone you could build WarpCTC like this:

```bash
mkdir build; cd build
cmake ..
make
```

Otherwise, set `WARP_CTC_PATH` to wherever you have `libwarpctc.so`
installed. If you have a GPU, you should also make sure that
`CUDA_HOME` is set to the home cuda directory (i.e. where
`include/cuda.h` and `lib/libcudart.so` live).

You should now be able to use `setup.py` to install the package into
your current Python environment:

```bash
python setup.py install
```

You can run a few unit tests with `setup.py` as well if you want:

```bash
python setup.py test
```

## Using the kernels

First import the module:

```python
import warpctc_tensorflow
```

The GPU kernel for the existing `CTCLoss` op is registered and ready
to use.  If you want to use WarpCTC as the CPU kernel for the
`CTCLoss` op you can use the ("experimental") `_kernel_label_map`
function to tell TensorFlow to use WarpCTC kernels instead of the
default CPU kernel:

```python
with tf.get_default_graph()._kernel_label_map({"CTCLoss": "WarpCTC"}):
    ...
    loss = tf.nn.ctc_loss(inputs, labels, seq_lens)
```

Note that `proprocess_collapse_repeated` must be `False` and
`ctc_merge_repeated` must be `True` (their default values) as these
options are not currently supported.

The WarpCTC op is available via the `warpctc_tensorflow.ctc` function:

```python
costs = warpctc_tensorflow.ctc(activations, flat_labels, label_lengths, input_lengths)
```

The `activations` input is a 3 dimensional Tensor and all the others
are single dimension Tensors.  See the main WarpCTC documentation for
more information.
    
    
