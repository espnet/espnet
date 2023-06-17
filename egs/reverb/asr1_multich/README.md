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

1. Training data: N-channel (N - default 2 - can be chosen by specifying the --nch-train parameter during executing the run.sh) simulation data from REVERB and clean data from WSJ (both WSJ0 and
WSJ1); Validation data: REVERB 8-channel real and simulation development
sets; 

    WSJ0 is single channel audio data and REVERB can be used with multi channels.
    So if single channel data comes in the mini-batch when training time,
    then skipping the neural network for the frontend part,
    and if multi channels comes from, passed it to the frontend (randomly passing WPE or BF).
    
1. Skipping the frontend part in the probability of 1/4 when training.

    To train the backend part with reverberant and noisy data,
    passing the multi-channel data to the backend part without the frontend,
    and the backend cannot handle multi-channel, so selecting 1-channel randomly.

1. Evaluation data: REVERB 8-channel real and simulation evaluation sets.

## Reference
- [1] *Neural network-based spectrum estimation for online WPE dereverberation; K. Kinoshita et al.. 2017;* http://www.kecl.ntt.co.jp/icl/signal/kinoshita/publications/Interspeech17/Neural%20Network-Based%20Spectrum%20Estimation%20for%20Online%20WPE%20Dereverberation.pdf
- [2] *Joint Optimization of Neural Network-based WPE Dereverberation and Acoustic Model for Robust Online ASR; J. Heymann et al.. 2019* https://ieeexplore.ieee.org/abstract/document/8683294
- [3] *Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;* https://dl.acm.org/citation.cfm?id=3305953
