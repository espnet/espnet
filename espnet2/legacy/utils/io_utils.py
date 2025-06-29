"""I/O Utils methods and classes."""

import io
import os

import h5py
import numpy as np
import soundfile


class SoundHDF5File(object):
    """Collecting sound files to a HDF5 file.

    >>> f = SoundHDF5File('a.flac.h5', mode='a')
    >>> array = np.random.randint(0, 100, 100, dtype=np.int16)
    >>> f['id'] = (array, 16000)
    >>> array, rate = f['id']


    :param: str filepath:
    :param: str mode:
    :param: str format: The type used when saving wav. flac, nist, htk, etc.
    :param: str dtype:

    """

    def __init__(self, filepath, mode="r+", format=None, dtype="int16", **kwargs):
        """Initialize Sound HDF5 File."""
        self.filepath = filepath
        self.mode = mode
        self.dtype = dtype

        self.file = h5py.File(filepath, mode, **kwargs)
        if format is None:
            # filepath = a.flac.h5 -> format = flac
            second_ext = os.path.splitext(os.path.splitext(filepath)[0])[1]
            format = second_ext[1:]
            if format.upper() not in soundfile.available_formats():
                # If not found, flac is selected
                format = "flac"

        # This format affects only saving
        self.format = format

    def __repr__(self):
        """Return class message."""
        return '<SoundHDF5 file "{}" (mode {}, format {}, type {})>'.format(
            self.filepath, self.mode, self.format, self.dtype
        )

    def create_dataset(self, name, shape=None, data=None, **kwds):
        """Create Dataset."""
        f = io.BytesIO()
        array, rate = data
        soundfile.write(f, array, rate, format=self.format)
        self.file.create_dataset(name, shape=shape, data=np.void(f.getvalue()), **kwds)

    def __setitem__(self, name, data):
        """Create Dataset for name and data."""
        self.create_dataset(name, data=data)

    def __getitem__(self, key):
        """Return item in given key."""
        data = self.file[key][()]
        f = io.BytesIO(data.tobytes())
        array, rate = soundfile.read(f, dtype=self.dtype)
        return array, rate

    def keys(self):
        """Return keys of files."""
        return self.file.keys()

    def values(self):
        """Return values of file key."""
        for k in self.file:
            yield self[k]

    def items(self):
        """Return keys and values of file key."""
        for k in self.file:
            yield k, self[k]

    def __iter__(self):
        """Iterate over items in file."""
        return iter(self.file)

    def __contains__(self, item):
        """Return items in file."""
        return item in self.file

    def __len__(self, item):
        """Return number of items in file."""
        return len(self.file)

    def __enter__(self):
        """Return self class."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file."""
        self.file.close()

    def close(self):
        """Close file."""
        self.file.close()
