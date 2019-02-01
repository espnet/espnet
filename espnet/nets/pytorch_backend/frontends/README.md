# [WIP] DNN based frontend system

[Warining] **Not all components are enough tested.**

## Requirements

- pytorch>=1.0
- python>=3.6
- https://github.com/kamo-naoyuki/pytorch_complex


## Components
All components are implemented using PyTorch, thus they can perform with its autograd mechanism.

1. DNN-Beamformer

    Implemented DNN MVDR based on 
    *Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;*
    https://arxiv.org/abs/1703.04783

1. DNN-WPE and conventional WPE

    Based on https://github.com/fgnt/nara_wpe

    The DNN based spectrogram estimation for WPE was proposed by
    *Neural network-based spectrum estimation for online WPE dereverberation; K. Kinoshita et al.. 2017;*
    https://pdfs.semanticscholar.org/f156/c1b463ad2b41c65220e09aa40e970be271d8.pdf

1. Feature transformation
    - Stft -> Fbank
    - Global/Utterance level mean  variance normalization

