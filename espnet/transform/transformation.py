import contextlib
from collections import OrderedDict
import copy
import importlib
import io
import json
import logging
import threading


def dynamic_import(import_path):
    alias = dict(
        delta='espnet.transform.add_deltas:AddDeltas',
        cmvn='espnet.transform.cmvn:CMVN',
        utterance_cmvn='espnet.transform.cmvn:UtteranceCMVN',
        fbank='espnet.transform.spectrogram:LogMelSpectrogram',
        spectrogram='espnet.transform.spectrogram:Spectrogram',
        stft='espnet.transform.spectrogram:Stft',
        wpe='espnet.transform.wpe:WPE',
        channel_selector='espnet.transform.channel_selector:ChannelSelector',
        )

    if import_path not in alias and ':' not in import_path:
        raise ValueError(
            'import_path should be one of {} or '
            'include ":", e.g. "espnet.transform.add_deltas:AddDeltas" : '
            '{}'.format(set(alias), import_path))
    if ':' not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(':')
    m = importlib.import_module(module_name)
    return getattr(m, objname)


class TransformConfig(object):
    def __init__(self, config=None, thread_local=True):
        if config is not None:
            if not isinstance(config, TransformConfig):
                raise TypeError('Must be {}, but got {}'
                                .format(self.__class__.__name__, type(config)))
            self.parent = config
        else:
            self.parent = {}

        if thread_local:
            self.status = threading.local().__dict__
        else:
            self.status = {}

    def __repr__(self):
        rep = ', '.join('{}={}'.format(k, v) for k, v in self.items())
        return '{}({})'.format(self.__class__.__name__,  rep)

    def __setitem__(self, key, value):
        self.status[key] = value

    def __getitem__(self, key):
        # Priority order: self > parent
        return self.status.get(key, self.parent[key])

    def __delitem__(self, key):
        del self.status[key]

    def __iter__(self):
        return iter(sorted(set(self.parent) | set(self.status)))

    def items(self):
        for k in self:
            yield k, self[k]

    def update(self, d=None, **kwargs):
        if d is not None:
            self.status.update(d, **kwargs)
        else:
            self.status.update(**kwargs)

    def get(self, key, default=None):
        return self.status.get(key, default=default)


global_transform_config = TransformConfig(thread_local=False)
global_transform_config.update(train=True)
transform_config = TransformConfig(global_transform_config,
                                   thread_local=True)


@contextlib.contextmanager
def using_transform_config(d, config=transform_config):
    assert isinstance(d, dict), type(d)
    assert isinstance(config, TransformConfig), type(config)

    old = {}
    for key, value in d.items():
        old[key] = config.get(key)
        config[key] = value
    try:
        yield config
    finally:
        for key, value in old.items():
            config[key] = value


class Transformation(object):
    """Apply some functions to the mini-batch

    Examples:
        >>> kwargs = {"process": [{"type": "fbank",
        ...                        "n_mels": 80,
        ...                        "fs": 16000},
        ...                       {"type": "cmvn",
        ...                        "stats": "data/train/cmvn.ark",
        ...                        "norm_vars": True},
        ...                       {"type": "delta", "window": 2, "order": 2}],
        ...           "mode": "sequential"}
        >>> transform = Transformation(**kwargs)
        >>> bs = 10
        >>> xs = [np.random.randn(100, 80).astype(np.float32)
        ...       for _ in range(bs)]
        >>> xs = transform(xs)
    """

    def __init__(self, conf=None, **kwargs):
        if conf is not None:
            with io.open(conf, encoding='utf-8') as f:
                conf = json.load(f)
                assert isinstance(conf, dict), type(conf)
            conf.update(kwargs)
            kwargs = conf

        if len(kwargs) == 0:
            self.conf = {'mode': 'sequential', 'process': []}
        else:
            # Deep-copy to avoid sharing of mutable objects
            self.conf = copy.deepcopy(kwargs)

        self.functions = OrderedDict()
        if self.conf.get('mode', 'sequential') == 'sequential':
            for idx, process in enumerate(self.conf['process']):
                assert isinstance(process, dict), type(process)
                opts = dict(process)
                process_type = opts.pop('type')
                class_obj = dynamic_import(process_type)
                self.functions[idx] = class_obj(**opts)
        else:
            raise NotImplementedError(
                'Not supporting mode={}'.format(self.conf['mode']))

    def __repr__(self):
        rep = '\n' + '\n'.join(
            '    {}: {}'.format(k, v) for k, v in self.functions.items())
        return '{}({})'.format(self.__class__.__name__, rep)

    def __call__(self, xs):
        """Return new mini-batch

        :param List[np.ndarray] xs:
        :return: batch:
        :rtype: List[np.ndarray]
        """
        if not isinstance(xs, (list, tuple)):
            is_batch = False
            xs = [xs]
        else:
            is_batch = True

        if self.conf.get('mode', 'sequential') == 'sequential':
            for idx in range(len(self.conf['process'])):
                func = self.functions[idx]
                try:
                    xs = [func(x) for x in xs]
                except Exception:
                    logging.fatal('Catch a exception from {}th func: {}'
                                  .format(idx, func))
                    raise
        else:
            raise NotImplementedError(
                'Not supporting mode={}'.format(self.conf['mode']))

        if is_batch:
            return xs
        else:
            return xs[0]
