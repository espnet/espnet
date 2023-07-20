"""Trainer module for speaker recognition."""
import argparse
import dataclasses
import logging
import time
from contextlib import contextmanager
from dataclasses import is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import humanfriendly
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.schedulers.abs_scheduler import (
    AbsBatchStepScheduler,
    AbsEpochStepScheduler,
    AbsScheduler,
    AbsValEpochStepScheduler,
)
from espnet2.torch_utils.add_gradient_noise import add_gradient_noise
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter, SubReporter
from espnet2.train.trainer import Trainer, TrainerOptions
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.eer import ComputeErrorRates, ComputeMinDcf, tuneThresholdfromScore
from espnet2.utils.kwargs2args import kwargs2args

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

autocast_args = dict()
if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import GradScaler, autocast

    if (
        V(torch.__version__) >= V("1.10.0")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        autocast_args = dict(dtype=torch.bfloat16)
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

    GradScaler = None

try:
    import fairscale
except ImportError:
    fairscale = None


class SpkTrainer(Trainer):
    """
    Trainer.
    Designed for speaker recognition.
    Training will be done as closed set classification.
    Validation will be open set EER calculation.

    """

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed

        model.eval()

        scores = []
        labels = []
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for utt_id, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch["utt_id"] = utt_id

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            org_shape = (batch["speech"].size(0), batch["speech"].size(1))
            batch["speech"] = batch["speech"].flatten(0, 1)
            batch["speech2"] = batch["speech2"].flatten(0, 1)

            speech_embds = model(
                speech=batch["speech"], spk_labels=None, extract_embd=True
            )
            speech2_embds = model(
                speech=batch["speech2"], spk_labels=None, extract_embd=True
            )

            speech_embds = F.normalize(speech_embds, p=2, dim=1)
            speech2_embds = F.normalize(speech2_embds, p=2, dim=1)

            speech_embds = speech_embds.view(org_shape[0], org_shape[1], -1)
            speech2_embds = speech2_embds.view(org_shape[0], org_shape[1], -1)

            for i in range(speech_embds.size(0)):
                score = torch.cdist(speech_embds[i], speech2_embds[i])
                score = -1.0 * torch.mean(score)
                scores.append(score.view(1))  # 0-dim to 1-dim tensor for cat
            labels.append(batch["spk_labels"])

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

        scores = torch.cat(scores).type(torch.float32)
        labels = torch.cat(labels).type(torch.int32).flatten()
        if distributed:
            # get the number of trials assigned on each GPU
            length = to_device(
                torch.tensor([labels.size(0)], dtype=torch.int32), "cuda"
            )
            lengths_all = [
                to_device(torch.zeros(1, dtype=torch.int32), "cuda")
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(lengths_all, length)

            scores_all = [
                to_device(torch.zeros(i, dtype=torch.float32), "cuda")
                for i in lengths_all
            ]
            torch.distributed.all_gather(scores_all, scores)
            scores = torch.cat(scores_all)

            labels_all = [
                to_device(torch.zeros(i, dtype=torch.int32), "cuda")
                for i in lengths_all
            ]
            torch.distributed.all_gather(labels_all, labels)
            labels = torch.cat(labels_all)
            rank = torch.distributed.get_rank()
            torch.distributed.barrier()
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        results = tuneThresholdfromScore(scores, labels, [1, 0.1])
        eer = results[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        p_trg, c_miss, c_fa = 0.05, 1, 1
        mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_trg, c_miss, c_fa)
        print("eer", eer, "mindcf", mindcf)

        reporter.register(stats=dict(eer=eer, mindcf=mindcf))
