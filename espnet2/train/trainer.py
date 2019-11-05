import argparse
import logging
from pathlib import Path
from typing import Union, Any

import numpy as np
import torch
import torch.optim
from pytypes import typechecked
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader

from espnet.asr.asr_utils import add_gradient_noise
from espnet2.train.build_scheduler import BatchScheduler, ValEpochScheduler
from espnet2.train.build_scheduler import build_batch_scheduler
from espnet2.train.build_scheduler import build_epoch_scheduler
from espnet2.train.build_optimizer import build_optimizer
from espnet2.train.build_scheduler import EpochScheduler
from espnet2.train.dataset import Dataset
from espnet2.train.reporter import Reporter


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


@typechecked
def get_parser(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()
    return parser


def train(model: torch.nn.Module, cmd=None) -> None:
    parser = get_parser()
    args = parser.parse_args(cmd)

    train_dataset = Dataset()
    train_batch_sampler = BatchSampler(shuffle=True)
    train_iter = DataLoader(dataset=train_dataset,
                            batch_sampler=train_batch_sampler)

    eval_dataset = Dataset()
    eval_batch_sampler = BatchSampler(shuffle=False)
    eval_iter = DataLoader(dataset=eval_dataset,
                           batch_sampler=eval_batch_sampler)

    optimizer = build_optimizer(args.optim, args.optim_arg)

    # batch_scheduler: invoked after every updating
    # e.g. torch.optim.lr_scheduler.CyclicLR
    if args.bscheduler is not None:
        batch_scheduler = build_batch_scheduler(args.bschedule, optimizer, args.bscheduler_arg)
    else:
        batch_scheduler = None

    # epoch_scheduler: invoked at every epochs
    # e.g. torch.optim.lr_scheduler.StepLR
    if args.escheduler is not None:
        epoch_scheduler = build_epoch_scheduler(args.eschedule, optimizer, args.escheduler_arg)
    else:
        epoch_scheduler = None
    reporter = Reporter(args.output_dir / 'report')

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    Trainer.load(model=model,
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

    Trainer.run(model=model,
                optimzer=optimizer,
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


class Trainer:
    # Don't define any instance methods in Trainer to avoid a God class
    @staticmethod
    @typechecked
    def load(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        reporter: Reporter,
        output_path: Union[str, Path],
        batch_scheduler: BatchScheduler = None,
        epoch_scheduler: EpochScheduler = None,
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
                batch_scheduler.load_state_dict(resumed_dict['batch_scheduler']) 
            if epoch_scheduler is not None:
                epoch_scheduler.load_state_dict(resumed_dict['epoch_scheduler']) 

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
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in state_dict}
            state_dict.update(pretrained_dict)
            obj.load_state_dict(state_dict)

    @staticmethod
    @typechecked
    def run(
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_iter,
            eval_iter,
            reporter: Reporter,
            output_path: Union[str, Path],
            batch_scheduler: BatchScheduler = None,
            epoch_scheduler: EpochScheduler = None,
            end_epoch: int = 30,
            patience: int = np.inf,
            ngpu: int = 1,
            train_dtype: str = 'float32',
            grad_noise: bool = False,
            accum_grad: int = 1,
            grad_clip_threshold: float = 0.05,
            ) -> None:
        # For apex supporting
        if train_dtype in ('O0', 'O1', 'O2', 'O3'):
            try:
                from apex import amp
            except ImportError:
                logging.error(f'You need to install apex. See https://github.com/NVIDIA/apex#linux')
                raise 
            model, optimizer = amp.initialize(model, optimizer, opt_level=train_dtype)
        if train_dtype in ('float16', 'float32', 'float64'):
            dtype = getattr(torch, train_dtype)
        else:
            dtype = torch.float32
        model = model.to(dtype=dtype, device='cuda' if ngpu > 0 else 'cpu')

        # Starting training process from here
        start_epoch = reporter.get_epoch()
        for iepoch in range(start_epoch + 1, end_epoch):
            reporter.set_epoch(iepoch)
            Trainer.train(model=model,
                          optimizer=optimizer,
                          iterator=train_iter,
                          reporter=reporter,
                          scheduler=batch_scheduler,
                          ngpu=ngpu,
                          use_apex=train_dtype in ('O0', 'O1', 'O2', 'O3'),
                          grad_noise=grad_noise,
                          accum_grad=accum_grad,
                          grad_clip_threshold=grad_clip_threshold
                          )
            Trainer.eval(model=model,
                         iterator=eval_iter,
                         reporter=reporter,
                         ngpu=ngpu)

            # Saves the best model
            for k, mode in [('loss', 'min'), ('acc', 'max')]:
                if reporter.has_key('eval', k) and reporter.best_epoch_and_value('eval', k, mode)[0] == iepoch:
                    torch.save(model.state_dict(), output_path / f'{k}.best.pt')
            # Saves snap shot for n-latest epochs
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'reporter': reporter.state_dict(),
                        'epoch_scheduler': epoch_scheduler.state_dict() if epoch_scheduler is not None else None,
                        'batch_scheduler': batch_scheduler.state_dict() if batch_scheduler is not None else None,
                        }, output_path / f'{k}epoch.pt')

            if epoch_scheduler is not None:
                # Controls opt-params by scheduler e.g. learning rate decay
                if isinstance(epoch_scheduler, ValEpochScheduler):
                    val = reporter.get_value('eval', 'acc' if reporter.has_key('eval', 'acc') else 'loss')
                    epoch_scheduler.step(val)
                else:
                    epoch_scheduler.step()

            # Early stopping
            if iepoch - reporter.best_epoch_and_value(
                    'eval', 
                    'acc' if reporter.has_key('eval', 'acc') else 'loss',
                    'max' if reporter.has_key('eval', 'acc') else 'min')[0] > patience:
                logging.info()
                break

    @staticmethod
    @typechecked
    def train(model: torch.nn.Module, 
              iterator, 
              optimizer: torch.nn.Optimizer,
              reporter: Reporter,
              scheduler: BatchScheduler = None,
              ngpu: int = 1,
              use_apex: bool = False,
              grad_noise: bool = False,
              accum_grad: int = 1,
              grad_clip_threshold: float = 0.05,
              ) -> None:
        model.train()

        for iiter, batch in enumerate(iterator, 1):
            batch = to_device(batch, 'cuda' if ngpu > 0 else 'cpu')
            if ngpu <= 1:
                loss, stats = model(**batch)
            else:
                loss, stats = data_parallel(model, range(ngpu), module_kwargs=batch)
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
                add_gradient_noise(model, reporter.total_count('train'), duration=100, eta=1.0, scale_factor=0.55)

            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_threshold)
            if iiter % accum_grad == 0:
                if torch.isnan(grad_norm):
                    logging.warning('The grad norm is nan. Skipping updating the model.')
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

    @staticmethod
    @typechecked
    @torch.no_grad()
    def eval(model: torch.nn.Module, iterator, reporter: Reporter,
             ngpu: int = 1) -> None:
        model.eval()
        for batch in iterator:
            batch = to_device(batch, 'cuda' if ngpu > 0 else 'cpu')
            if ngpu <= 1:
                _, stats = model(**batch)
            else:
                _, stats = data_parallel(model, range(ngpu), module_kwargs=batch)
                stats = {k: v.mean(0).item() for k, v in stats.items()}

            reporter.report('eval', stats)
