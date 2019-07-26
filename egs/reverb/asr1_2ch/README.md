# [WIP] DNN based frontend system

[Warining] **Not all components are enough tested.**

The recipe for REVERB is based on the CHiME-4 multi-channel recipe: https://github.com/espnet/espnet/tree/master/egs/chime4/asr1_multich,
but the strategy for training and test is slightly different.

## Components

1. DNN-WPE and conventional WPE: [1], [2]

    Based on https://github.com/nttcslab-sp/dnn_wpe

1. DNN-Beamformer: [3]

    Based on https://github.com/fgnt/nn-gev

1. Feature transformation
    - Stft -> Fbank
    - Global/Utterance level mean variance normalization


## The training and test strategy

1. Using REVERB 2-ch data and WSJ0 data as training set.

    WSJ0 is single channel audio data and REVERB can be used with multi channels.
    So if single channel data comes in the mini-batch when training time,
    then skipping the neural network for the frontend part,
    and if multi channels comes from, passed it to the frontend.

    Note that different channel data are created to separated mini-batchs.
    To do this, these two data are given with different "category" tags.
    
1. Skipping the frontend part in the probability of 1/2 when training.

    To train the backend part with reverberant and noisy data,
    passing the multi-channel data to the backend part without the frontend,
    and the backend cannot handle multi-channel, so selecting 1-channel randomly.

1. Test with 8-ch simulation and real data
    
    

## Reference
- [1] *Neural network-based spectrum estimation for online WPE dereverberation; K. Kinoshita et al.. 2017;* http://www.kecl.ntt.co.jp/icl/signal/kinoshita/publications/Interspeech17/Neural%20Network-Based%20Spectrum%20Estimation%20for%20Online%20WPE%20Dereverberation.pdf
- [2] *Joint Optimization of Neural Network-based WPE Dereverberation and Acoustic Model for Robust Online ASR; J. Heymann et al.. 2019* https://ieeexplore.ieee.org/abstract/document/8683294
- [3] *Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;* https://dl.acm.org/citation.cfm?id=3305953
