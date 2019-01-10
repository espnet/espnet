import librosa
import numpy as np


def spectrogram(x, n_fft, n_shift,
                win_length=None, window='hann'):
    spc = np.abs(librosa.stft(x, n_fft, n_shift, win_length, window=window)).T
    return spc


class Spectrogram(object):
    def __init__(self, n_fft, n_shift, win_length=None, window='hann'):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window

    def __repr__(self):
        return ('{name}(n_fft={n_fft}, n_shift={n_shift}, '
                'win_length={win_length}, window={window})'
                .format(name=self.__class__.__name__,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        window=self.window))

    def __call__(self, x):
        return spectrogram(x.astype(np.float32),
                           n_fft=self.n_fft, n_shift=self.n_shift,
                           win_length=self.win_length,
                           window=self.window)


def logmelspectrogram(x, fs, n_mels, n_fft, n_shift,
                      win_length=None, window='hann', fmin=None, fmax=None,
                      eps=1e-10):
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(fs, n_fft, n_mels, fmin, fmax)
    spc = np.abs(librosa.stft(x, n_fft, n_shift, win_length, window=window))
    lmspc = np.log10(np.maximum(eps, np.dot(mel_basis, spc).T))

    return lmspc


class LogMelSpectrogram(object):
    def __init__(self, fs, n_mels, n_fft, n_shift, win_length=None, window='hann',
                 fmin=None, fmax=None, eps=1e-10):
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps

    def __repr__(self):
        return ('{name}(fs={fs}, n_mels={n_mels}, n_fft={n_fft}, '
                'n_shift={n_shift}, win_length={win_length}, window={window}, '
                'fmin={fmin}), fmax={fmax}, eps={eps}))'
                .format(name=self.__class__.__name__,
                        fs=self.fs,
                        n_mels=self.n_mels,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        window=self.window,
                        fmin=self.fmin,
                        fmax=self.fmax,
                        eps=self.eps))

    def __call__(self, x):
        return logmelspectrogram(
            x.astype(np.float32),
            fs=self.fs,
            n_mels=self.n_mels,
            n_fft=self.n_fft, n_shift=self.n_shift,
            win_length=self.win_length,
            window=self.window)


class Stft(object):
    def __init__(self, n_fft, n_shift, win_length=None,
                 window='hann', center=True, pad_mode='reflect'):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def __repr__(self):
        return ('{name}(n_fft={n_fft}, n_shift={n_shift}, '
                'win_length={win_length}, window={window},'
                'center={center}, pad_mode={pad_mode})'
                .format(name=self.__class__.__name__,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        window=self.window,
                        center=self.center,
                        pad_mode=self.pad_mode))

    def __call__(self, x):
        # x: [Time, Channel]
        if x.ndim == 1:
            single_channel = True
            # x: [Time] -> [Time, Channel]
            x = x[:, None]
            channel = 1
        else:
            single_channel = False
            channel = x.shape[1]
        x = x.astype(np.float32)

        # FIXME(kamo): librosa.stft can't use multi-channel?
        # x: [Time, Channel, Freq]
        x = np.stack([librosa.stft(
            x=x[:, ch],
            n_fft=self.n_fft,
            hop_length=self.n_shift,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode).T
            for ch in range(channel)], axis=1)
        if single_channel:
            # x: [Time, Channel, Freq] -> [Time, Freq]
            x = x[:, 0]
        return x
