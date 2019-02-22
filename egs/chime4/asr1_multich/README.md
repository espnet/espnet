# [WIP](kamo) DNN based frontend system

[Warining] **Not all components are enough tested.**

## Requirements

- pytorch>=1.0
- python>=3.6
- https://github.com/kamo-naoyuki/pytorch_complex


## Components
All components are implemented using PyTorch, thus they can perform with its autograd mechanism.

1. DNN-WPE and conventional WPE

    Based on https://github.com/fgnt/nara_wpe

    The DNN based spectrogram estimation for WPE was proposed by
    *Neural network-based spectrum estimation for online WPE dereverberation; K. Kinoshita et al.. 2017;*
    https://pdfs.semanticscholar.org/f156/c1b463ad2b41c65220e09aa40e970be271d8.pdf

    Note that DNN-WPE didn't result significant gains comparing with offline-WPE.

1. DNN-Beamformer

    Implemented DNN MVDR based on
    *Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;*
    https://arxiv.org/abs/1703.04783

    The codes is based on https://github.com/fgnt/nn-gev

    There are also gev implementation, but it can't work now
    because Eigen value decomposition is not implemented.

1. Feature transformation
    - Stft -> Fbank
    - Global/Utterance level mean variance normalization


## The traning strategy

1. Using Chime4 data and WSJ0 data as training set.
    WSJ0 is single channel audio data and Chime4 can be used with multi channels.
    so If single channel data comes in the mini-batch when traning time,
    then skipping the neural network for the frontend part,
    and if multi channels comes from, passed it to the frontend.

    Note that different channel data are created to separated mini-batchs.
    To do this, these two data are given with different "category" tags.
1. Skipping the frontend part in the probability of 1/2 when training.
    To traing the backend part with nolsy data,
    passing the multi-channel data to the backend part without the frontend,
    and the backend cannot handle multi-channel, so selecting 1-channel randomly.
