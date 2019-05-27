#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from espnet.asr import batchfy


def make_batchset(data, batch_size, max_length_in, max_length_out,
                  num_batches=0, batch_sort_key='shuffle', min_batch_size=1, shortest_first=False,
                  count="auto", batch_bins=0, batch_frames_in=0, batch_frames_out=0, batch_frames_inout=0):
    """Make batch set from json dictionary

    if utts have "category" value,

        >>> data = {'utt1': {'category': 'A', 'input': ...},
        ...         'utt2': {'category': 'B', 'input': ...},
        ...         'utt3': {'category': 'B', 'input': ...},
        ...         'utt4': {'category': 'A', 'input': ...}}
        >>> make_batchset(data, batchsize=2, ...)
        [[('utt1', ...), ('utt4', ...)], [('utt2', ...), ('utt3': ...)]]

    Note that if any utts doesn't have "category",
    perform as same as batchfy_by_{count}

    :param Dict[str, Dict[str, Any]] data: dictionary loaded from data.json
    :param int batch_size: maximum number of sequences in a minibatch.
    :param int batch_bins: maximum number of bins (frames x dim) in a minibatch.
    :param int batch_frames_in:  maximum number of input frames in a minibatch.
    :param int batch_frames_out: maximum number of output frames in a minibatch.
    :param int batch_frames_out: maximum number of input+output frames in a minibatch.
    :param str count: strategy to count maximum size of batch.
        For choices, see espnet.asr.batchfy.BATCH_COUNT_CHOICES

    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int num_batches: # number of batches to use (for debug)
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse
        :return: List[List[Tuple[str, dict]]] list of batches

    NOTE: this function just reverses `batch_sort_key` of "input" to "output" or vise versa.
    """
    return batchfy.make_batchset(
        data=data,
        batch_size=batch_size,
        max_length_in=max_length_in,
        max_length_out=max_length_out,
        num_batches=num_batches,
        batch_sort_key=batch_sort_key,
        min_batch_size=min_batch_size,
        shortest_first=shortest_first,
        count=count,
        batch_bins=batch_bins,
        batch_frames_in=batch_frames_in,
        batch_frames_out=batch_frames_out,
        batch_frames_inout=batch_frames_inout,
        swap_io=True
    )
