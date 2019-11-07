import argparse
import logging
import random
import sys
from abc import ABC, abstractmethod
from distutils.util import strtobool
from pathlib import Path
from typing import Optional, Union, Any

import configargparse
import numpy as np
import torch
import torch.nn
import torch.optim
import yaml
from pytypes import typechecked
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader

from espnet.asr.asr_utils import add_gradient_noise
from espnet2.train.build_optimizer import build_optimizer
from espnet2.train.build_scheduler import (
    build_batch_scheduler, AbsEpochScheduler, AbsBatchScheduler,
    build_epoch_scheduler, AbsValEpochScheduler, )
from espnet2.train.dataset import Dataset, BatchSampler, collate_fn
from espnet2.train.reporter import Reporter


class BaseTask(ABC):
    # Use @staticmethod, or @classmethod,
    # instead of instance method to avoid God classes

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @typechecked
    @classmethod
    def add_arguments(cls, parser: configargparse.ArgumentParser = None) \
            -> argparse.ArgumentParser:
        if parser is None:
            parser = configargparse.ArgumentParser(description='base parser')

        def dummy(arg):
            raise TypeError

        # Note(kamo): Use '_' instead of '-' to avoid confusion for separator

        group = parser.add_argument_group('The common configuration')

        group.add_argument('--config', is_config_file=True,
                           help='config file path')
        group.add_argument('--log_level', type=str, default='INFO',
                           choices=['INFO', 'ERROR', 'WARNING', 'INFO',
                                    'DEBUG', 'NOTSET'])

        group.add_argument('--gen_yaml', type=str,
                           help='Generate a default yaml file to '
                                'the specified path for this task')
        group.add_argument('--output_dir', type=str, default='output')
        group.add_argument('--ngpu', type=int)
        group.add_argument('--seed', type=int, default=0)

        group.add_argument('--resume_epoch', type=int)
        group.add_argument('--resume_path', type=str)
        group.add_argument('--preatrain_path', type=int)
        group.add_argument('--pretrain_key', type=str)

        group.add_argument('--train_dtype', type=str, default='float32')
        group.add_argument('--patience', type=int, default=None)
        group.add_argument('--batchsize', type=int, default=20)
        group.add_argument('--grad_noise', type=strtobool, default=False)
        group.add_argument('--accum_grad', type=int, default=1)
        group.add_argument('--grad_clip_threshold', type=float, default=1e-4)

        group.add_argument('--train_dataset_config', type=dummy,
                           default=dict())
        group.add_argument('--train_batch_config', type=dummy,
                           default=dict())
        group.add_argument('--eval_dataset_config', type=dummy,
                           default=dict())
        group.add_argument('--eval_batch_config', type=dummy,
                           default=dict())

        group.add_argument('--optimizer', type=str, default='adam')
        group.add_argument('--optimizer_arg', type=dummy,
                           default=dict(lr=1e-3))
        group.add_argument('--escheduler', type=str)
        group.add_argument('--escheduler_arg', type=dummy,
                           default=dict())
        group.add_argument('--bscheduler', type=str)
        group.add_argument('--bscheduler_arg', type=dummy,
                           default=dict())

        return parser

    @typechecked
    @classmethod
    def build_optimizer(cls, model: torch.nn.Module,
                        args: argparse.Namespace) -> torch.optim.Optimizer:
        return build_optimizer(model=model, optim=args.optim,
                               kwarg=args.optim_arg)

    @typechecked
    @classmethod
    def build_epoch_scheduler(cls, optimizer, args: argparse.Namespace) \
            -> Optional[AbsEpochScheduler]:
        return build_epoch_scheduler(
            optimizer=optimizer, scheduler=args.escheduler,
            kwargs=args.escheduler_arg)

    @typechecked
    @classmethod
    def build_batch_scheduler(cls, optimizer, args: argparse.Namespace) \
            -> Optional[AbsBatchScheduler]:
        return build_batch_scheduler(
            optimizer=optimizer, scheduler=args.bscheduler,
            kwargs=args.bscheduler_arg)

    @typechecked
    @classmethod
    @abstractmethod
    def build_model(cls, idim: int, odim: int, args: argparse.Namespace):
        raise NotImplementedError

    @typechecked
    @classmethod
    @abstractmethod
    def get_default_config(cls) -> dict:
        raise NotImplementedError

    @typechecked
    @classmethod
    def main(cls, args: argparse.Namespace = None, cmd: str = None) -> None:
        if args is None:
            parser = cls.add_arguments()
            args = parser.parse_args(cmd)

        logging.basicConfig(
            level=args.log_level,
            format=
            '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

        if args.gen_yaml is not None:
            config = cls.get_default_config()
            p = Path(args.gen_yaml)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w') as f:
                yaml.dump(config, f, Dumper=yaml.Dumper)
            logging.info('Yaml file was generated: {p}')
            sys.exit(0)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)

        train_dataset = Dataset(args.train_dataset_config)
        train_batch_sampler = BatchSampler(config=args.train_batch_config,
                                           shuffle=True)
        train_iter = DataLoader(dataset=train_dataset,
                                batch_sampler=train_batch_sampler,
                                collate_fn=collate_fn)

        eval_dataset = Dataset(args.eval_dataset_config)
        eval_batch_sampler = BatchSampler(config=args.eval_batch_config,
                                          shuffle=False)
        eval_iter = DataLoader(dataset=eval_dataset,
                               batch_sampler=eval_batch_sampler,
                               collate_fn=collate_fn)

        model = cls.build_model()
        optimizer = cls.build_optimizer(model=model, args=args)

        # batch_scheduler: invoked after every updating
        # e.g. torch.optim.lr_scheduler.CyclicLR
        batch_scheduler = cls.build_batch_scheduler(
            optimizer=optimizer, args=args)

        # epoch_scheduler: invoked at every epochs
        # e.g. torch.optim.lr_scheduler.StepLR
        epoch_scheduler = cls.build_epoch_scheduler(
            optimizer=optimizer, args=args)
        reporter = Reporter(args.output_dir / 'report')

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cls.load(model=model,
                 optimizer=optimizer,
                 reporter=reporter,
                 output_path=output_path,
                 batch_scheduler=batch_scheduler,
                 epoch_scheduler=epoch_scheduler,
                 resume_epoch=args.resume_epoch,
                 resume_path=args.resume_path,
                 pretrain_path=args.pretrain_path,
                 pretrain_key=args.pretrain_key,
                 )

        cls.run(model=model,
                optimizer=optimizer,
                train_iter=train_iter,
                eval_iter=eval_iter,
                reporter=reporter,
                output_path=output_path,
                batch_scheduler=batch_scheduler,
                epoch_scheduler=epoch_scheduler,
                ngpu=args.ngpu,
                train_dtype=args.train_dtype,
                patience=args.patience,
                grad_noise=args.grad_noise,
                accum_grad=args.accum_grad,
                grad_clip_threshold=args.grad_clip_threshold
                )

    @typechecked
    @classmethod
    def load(cls,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             reporter: Reporter,
             output_path: Union[str, Path],
             batch_scheduler: AbsBatchScheduler = None,
             epoch_scheduler: AbsEpochScheduler = None,
             resume_epoch: int = None,
             resume_path: Union[str, Path] = None,
             pretrain_path: Union[str, Path] = None,
             pretrain_key: str = None) -> None:
        # For resuming: Specify either resume_epoch or resume_path.
        #     - resume_epoch: Load from outdir/{}epoch.pt.
        #     - resume_path: Load from the specified path.
        # Find the latest epoch snapshot
        if resume_epoch == 'latest':
            resume_epoch = 0
            for p in output_path.glob('*epoch.pt'):
                try:
                    n = int(p.stem.replace('epoch', ''))
                except TypeError:
                    continue
                if n > resume_epoch:
                    resume_epoch = n
            # If not found any snapshots, then nothing is done
            if resume_epoch == 0:
                resume_epoch = None
        if resume_epoch is not None or resume_path is not None:
            if resume_path is not None:
                resume_path = output_path / f'{resume_epoch}epoch.pt'
            resumed_dict = torch.load(resume_path)
            model.load_state_dict(resumed_dict['model'])
            optimizer.load_state_dict(resumed_dict['optimizer'])
            reporter.load_state_dict(resumed_dict['reporter'])
            if batch_scheduler is not None:
                batch_scheduler.load_state_dict(
                    resumed_dict['batch_scheduler'])
            if epoch_scheduler is not None:
                epoch_scheduler.load_state_dict(
                    resumed_dict['epoch_scheduler'])

        # For distillation, fine-tuning, transfer learning, etc.
        if pretrain_path is not None:
            if pretrain_key is None:
                obj = model
            else:
                def get_attr(obj: Any, key: str):
                    """

                    >>> class A(torch.nn.Module):
                    ...     def __init__(self):
                    ...         super().__init__()
                    ...         self.linear = torch.nn.Linear(10, 10)
                    >>> a = A()
                    >>> assert A.linear.weight is get_attr(A, 'linear.weight')

                    """
                    if key.strip() == '':
                        return obj
                    for k in key.split('.'):
                        obj = getattr(obj, k)
                    return obj

                obj = get_attr(model, pretrain_key)

            state_dict = obj.state_dict()
            pretrained_dict = torch.load(pretrain_path)
            # Ignores the parameters not existing in the train-model
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in state_dict}
            state_dict.update(pretrained_dict)
            obj.load_state_dict(state_dict)

    @typechecked
    @classmethod
    def run(cls,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_iter,
            eval_iter,
            reporter: Reporter,
            output_path: Union[str, Path],
            batch_scheduler: AbsBatchScheduler = None,
            epoch_scheduler: AbsEpochScheduler = None,
            end_epoch: int = 30,
            patience: int = np.inf,
            ngpu: int = 1,
            train_dtype: str = 'float32',
            grad_noise: bool = False,
            accum_grad: int = 1,
            grad_clip_threshold: float = 0.05) -> None:
        # For apex supporting
        if train_dtype in ('O0', 'O1', 'O2', 'O3'):
            try:
                from apex import amp
            except ImportError:
                logging.error(f'You need to install apex. '
                              f'See https://github.com/NVIDIA/apex#linux')
                raise
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level=train_dtype)
        if train_dtype in ('float16', 'float32', 'float64'):
            dtype = getattr(torch, train_dtype)
        else:
            dtype = torch.float32
        model = model.to(dtype=dtype, device='cuda' if ngpu > 0 else 'cpu')

        # Starting training process from here
        start_epoch = reporter.get_epoch()
        for iepoch in range(start_epoch + 1, end_epoch):
            reporter.set_epoch(iepoch)
            cls.train(model=model,
                      optimizer=optimizer,
                      iterator=train_iter,
                      reporter=reporter,
                      scheduler=batch_scheduler,
                      ngpu=ngpu,
                      use_apex=train_dtype in ('O0','O1','O2','O3'),
                      grad_noise=grad_noise,
                      accum_grad=accum_grad,
                      grad_clip_threshold=grad_clip_threshold
                      )
            cls.eval(model=model,
                     iterator=eval_iter,
                     reporter=reporter,
                     ngpu=ngpu)
            reporter.plot()

            # Saves the best model
            for k, mode in [('loss', 'min'), ('acc', 'max')]:
                best_epoch, _ = reporter.best_epoch_and_value('eval', k, mode)
                if reporter.has_key('eval', k) and best_epoch == iepoch:
                    torch.save(
                        model.state_dict(), output_path / f'{k}.best.pt')
            # Saves snap shot for n-latest epochs
            torch.save(
                {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'reporter': reporter.state_dict(),
                 'epoch_scheduler': epoch_scheduler.state_dict()
                 if epoch_scheduler is not None else None,
                 'batch_scheduler': batch_scheduler.state_dict()
                 if batch_scheduler is not None else None,
                 }, output_path / f'{k}epoch.pt')

            if epoch_scheduler is not None:
                # Controls opt-params by scheduler e.g. learning rate decay
                if isinstance(epoch_scheduler, AbsValEpochScheduler):
                    val = reporter.get_value(
                        'eval', 'acc'
                        if reporter.has_key('eval', 'acc') else 'loss')
                    epoch_scheduler.step(val)
                else:
                    epoch_scheduler.step()

            # Early stopping
            best_epoch, _ = reporter.best_epoch_and_value(
                'eval',
                'acc' if reporter.has_key('eval', 'acc') else 'loss',
                'max' if reporter.has_key('eval', 'acc') else 'min')
            if patience is not None and iepoch - best_epoch > patience:
                logging.info()
                break

    @typechecked
    @classmethod
    def train(cls,
              model: torch.nn.Module,
              iterator,
              optimizer: torch.optim.Optimizer,
              reporter: Reporter,
              scheduler: AbsBatchScheduler = None,
              ngpu: int = 1,
              use_apex: bool = False,
              grad_noise: bool = False,
              accum_grad: int = 1,
              grad_clip_threshold: float = 0.05,
              ) -> None:
        model.train()

        for iiter, batch in enumerate(iterator, 1):
            assert isinstance(batch, dict), type(batch)
            batch = to_device(batch, 'cuda' if ngpu > 0 else 'cpu')
            if ngpu <= 1:
                loss, stats = model(**batch)
            else:
                loss, stats = data_parallel(model, range(ngpu),
                                            module_kwargs=batch)
                loss = loss.mean(0)
                stats = {k: v.mean(0) for k, v in stats.items()}

            reporter.report('train', stats)
            if use_apex:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            loss.detach_()

            # gradient noise injection
            if grad_noise:
                add_gradient_noise(model, reporter.total_count('train'),
                                   duration=100, eta=1.0, scale_factor=0.55)

            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_threshold)
            if iiter % accum_grad == 0:
                if torch.isnan(grad_norm):
                    logging.warning(
                        'The grad norm is nan. Skipping updating the model.')
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

    @torch.no_grad()
    @typechecked
    @classmethod
    def eval(cls, model: torch.nn.Module, iterator, reporter: Reporter,
             ngpu: int = 1) -> None:
        model.eval()
        for batch in iterator:
            assert isinstance(batch, dict), type(batch)
            batch = to_device(batch, 'cuda' if ngpu > 0 else 'cpu')
            if ngpu <= 1:
                _, stats = model(**batch)
            else:
                _, stats = data_parallel(model, range(ngpu),
                                         module_kwargs=batch)
                stats = {k: v.mean(0).item() for k, v in stats.items()}

            reporter.report('eval', stats)


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

