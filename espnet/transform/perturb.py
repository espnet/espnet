import numpy
import soundfile

from espnet.utils.io_utils import SoundHDF5File

# TODO(kamo): Please implement, if anyone is interesting


class SpeedPerturbation(object):
    # The marker used by "Transformation"
    accept_uttid = False

    def __init__(self, lower=0.8, upper=1.2, utt2scale=None):
        self.utt2scale = {}

        if utt2scale is not None:
            self.utt2scale_file = utt2scale
            self.lower = None
            self.upper = None
            self.accept_uttid = True

            with open(utt2scale, 'r') as f:
                for line in f:
                    utt, scale = line.rstrip().split(None, 1)
                    scale = float(scale)
                    self.utt2scale[utt] = scale
        else:
            self.lower = lower
            self.upper = upper

    def __repr__(self):
        if len(self.utt2scale) == 0:
            return '{}({lower={}, upper={})'.format(
                self.__class__.__name__, self.lower, self.upper)
        else:
            return '{}({})'.format(self.__class__.__name__,
                                   self.utt2scale_file)

    def __call__(self, x, uttid=None):
        # if self.accept_uttid:
        #     scale = self.utt2scale[uttid]
        # else:
        #     scale = numpy.random.uniform(self.lower, self.upper)
        raise NotImplementedError


class VolumePerturbation(object):
    # The marker used by "Transformation"
    accept_uttid = False

    def __init__(self, lower=0.8, upper=1.2, utt2scale=None):
        self.utt2scale = {}

        if utt2scale is not None:
            self.utt2scale_file = utt2scale
            self.lower = None
            self.upper = None
            self.accept_uttid = True

            with open(utt2scale, 'r') as f:
                for line in f:
                    utt, scale = line.rstrip().split(None, 1)
                    scale = float(scale)
                    self.utt2scale[utt] = scale
        else:
            self.lower = lower
            self.upper = upper

    def __repr__(self):
        if len(self.utt2scale) == 0:
            return '{}({lower={}, upper={})'.format(
                self.__class__.__name__, self.lower, self.upper)
        else:
            return '{}({})'.format(self.__class__.__name__,
                                   self.utt2scale_file)

    def __call__(self, x, uttid=None):
        if self.accept_uttid:
            scale = self.utt2scale[uttid]
        else:
            scale = numpy.random.uniform(self.lower, self.upper)
        return x * scale


class NoiseInjection(object):
    # The marker used by "Transformation"
    accept_uttid = True

    def __init__(self, utt2noise, utt2snr, filetype='list'):
        self.utt2noise_file = utt2noise
        self.utt2snr_file = utt2snr
        self.filetype = filetype

        self.utt2snr = {}
        with open(utt2noise, 'r') as f:
            for line in f:
                utt, snr = line.rstrip().split(None, 1)
                snr = float(snr)
                self.utt2snr[utt] = snr

        self.utt2noise = {}
        if filetype == 'list':
            with open(utt2noise, 'r') as f:
                for line in f:
                    utt, filename = line.rstrip().split(None, 1)
                    signal, rate = soundfile.read(filename, dtype='int16')
                    self.utt2noise[utt] = (signal, rate)

        elif filetype == 'sound.hdf5':
            self.utt2noise = SoundHDF5File(utt2noise, 'r')
        else:
            raise ValueError(filetype)

        if set(self.utt2snr) != set(self.utt2noise):
            raise RuntimeError('The uttids mismatch between {} and {}'
                               .format(utt2snr, utt2noise))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               self.utt2noise_file)

    def __call__(self, x, uttid):
        # noise, rate self.utt2noise[uttid]
        # snr = self.utt2snr[uttid]
        raise NotImplementedError


class RIRConvolve(object):
    # The marker used by "Transformation"
    accept_uttid = True

    def __init__(self, utt2rir, filetype='list'):
        self.utt2rir_file = utt2rir
        self.filetype = filetype

        self.utt2rir = {}
        if filetype == 'list':
            with open(utt2rir, 'r') as f:
                for line in f:
                    utt, filename = line.rstrip().split(None, 1)
                    signal, rate = soundfile.read(filename, dtype='int16')
                    self.utt2rir[utt] = (signal, rate)

        elif filetype == 'sound.hdf5':
            self.utt2rir = SoundHDF5File(utt2rir, 'r')
        else:
            raise NotImplementedError(filetype)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               self.utt2rir_file)

    def __call__(self, x, uttid):
        # rir, rate = self.utt2rir[uttid]
        raise NotImplementedError
