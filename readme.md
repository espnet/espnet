Fast CTC for Chainer
====================

This module contains two implementations of CTC [Graves2006] for [Chainer](http://chainer.org).

    1. Is build using custom CUDA kernels / Cython code
    2. Is a wrapper around Baidu's warp-ctc
    
The first one is faster than the original Chainer implementation but slower than warp-ctc.
However, it allows for easier modification of the CTC algorithm.
If you want to tinker with it, use the first method. If you just want speed, use warp-ctc.

Requirements
------------

    1. Chainer
    2. Python 3.x
    3. GPU + CUDA for high performance

Installation
------------

First, clone this repository.
```
git clone https://github.com/jheymann85/chainer_ctc.git
```

Next, we need to install warp-ctc. This can be done with the install 
script

```
chmod +x install_warp-ctc.sh
./install_warp-ctc.sh
```

Now you can install the module

```
pip install --user -e .
```

Finally, run the tests to see if the installation was successful.


Credits
-------

Baidu for their incredible fast [warp-ctc](https://github.com/baidu-research/warp-ctc) implementation.

\[Graves2006\]: Alex Graves, Santiago Fernandez, Faustino Gomez, Jurgen Schmidhuber
    **Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks**
    <ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf>