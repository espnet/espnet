import logging
import os
import shutil
import tempfile

import torch

import chainer
from chainer.datasets import TransformDataset
from chainer.serializers.npz import DictionarySerializer
from chainer.serializers.npz import NpzDeserializer
from chainer.training import extension

from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator


def torch_save(path, model):
    """Function to save torch model states

    :param str path: file path to be saved
    :param torch.nn.Module model: torch model
    """
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def torch_load(path, model):
    """Function to load torch model states

    :param str path: model file or snapshot file to be loaded
    :param torch.nn.Module model: torch model
    """
    if 'snapshot' in path:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']
    else:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict


def torch_resume(snapshot_path, trainer):
    """Function to resume from snapshot for pytorch

    :param str snapshot_path: snapshot file path
    :param instance trainer: chainer trainer instance
    """
    # load snapshot
    snapshot_dict = torch.load(snapshot_path, map_location=lambda storage, loc: storage)

    # restore trainer states
    d = NpzDeserializer(snapshot_dict['trainer'])
    d.load(trainer)

    # restore model states
    if hasattr(trainer.updater.model, "model"):
        # (for TTS model)
        if hasattr(trainer.updater.model.model, "module"):
            trainer.updater.model.model.module.load_state_dict(snapshot_dict['model'])
        else:
            trainer.updater.model.model.load_state_dict(snapshot_dict['model'])
    else:
        # (for ASR model)
        if hasattr(trainer.updater.model, "module"):
            trainer.updater.model.module.load_state_dict(snapshot_dict['model'])
        else:
            trainer.updater.model.load_state_dict(snapshot_dict['model'])

    # retore optimizer states
    trainer.updater.get_optimizer('main').load_state_dict(snapshot_dict['optimizer'])

    # delete opened snapshot
    del snapshot_dict


def torch_snapshot(savefun=torch.save,
                   filename='snapshot.ep.{.updater.epoch}'):
    """Returns a trainer extension to take snapshots of the trainer for pytorch."""

    @extension.make_extension(trigger=(1, 'epoch'), priority=-100)
    def torch_snapshot(trainer):
        _torch_snapshot_object(trainer, filename.format(trainer), savefun)

    return torch_snapshot


def _torch_snapshot_object(trainer, filename, savefun):
    # make snapshot_dict dictionary
    s = DictionarySerializer()
    s.save(trainer)
    if hasattr(trainer.updater.model, "model"):
        # (for TTS)
        if hasattr(trainer.updater.model.model, "module"):
            model_state_dict = trainer.updater.model.model.module.state_dict()
        else:
            model_state_dict = trainer.updater.model.model.state_dict()
    else:
        # (for ASR)
        if hasattr(trainer.updater.model, "module"):
            model_state_dict = trainer.updater.model.module.state_dict()
        else:
            model_state_dict = trainer.updater.model.state_dict()
    snapshot_dict = {
        "trainer": s.target,
        "model": model_state_dict,
        "optimizer": trainer.updater.get_optimizer('main').state_dict()
    }

    # save snapshot dictionary
    fn = filename.format(trainer)
    prefix = 'tmp' + fn
    tmpdir = tempfile.mkdtemp(prefix=prefix, dir=trainer.out)
    tmppath = os.path.join(tmpdir, fn)
    try:
        savefun(snapshot_dict, tmppath)
        shutil.move(tmppath, os.path.join(trainer.out, fn))
    finally:
        shutil.rmtree(tmpdir)


def warn_if_no_cuda():
    """Emits a warning if cuda is not available"""
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')


def get_iterators(train, valid, converter, n_iter_processes, use_sortagrad=False):
    """Returns training and validation iterators

    :param train: The training data
    :param valid: The validation data
    :param converter: The batch converter
    :param int n_iter_processes: The number of iterating processes
    :return: (train_iter, valid_iter)
    """
    # hack to make batchsize argument as 1
    # actual batchsize is included in a list
    if n_iter_processes > 0:
        train_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(train, converter.transform),
            batch_size=1, shuffle=not use_sortagrad, n_processes=n_iter_processes, n_prefetch=8, maxtasksperchild=20)
        valid_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(valid, converter.transform),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        train_iter = ToggleableShufflingSerialIterator(
            TransformDataset(train, converter.transform),
            batch_size=1, shuffle=not use_sortagrad)
        valid_iter = ToggleableShufflingSerialIterator(
            TransformDataset(valid, converter.transform),
            batch_size=1, repeat=False, shuffle=False)
    return train_iter, valid_iter
