
""" DeepSpeed Trainer Module """

import torch
import deepspeed
import logging
import time
import json
import argparse
import dataclasses

from deepspeed import DeepSpeedEngine
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from typeguard import typechecked
from torch.distributed import ReduceOp

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.trainer import Trainer
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.train.reporter import Reporter, SubReporter


@dataclasses.dataclass
class DeepSpeedTrainerOptions:
    resume: bool
    seed: int
    train_dtype: Union[str, torch.dtype]
    log_interval: Optional[int]
    output_dir: Union[Path, str]
    max_epoch: int
    patience: Optional[int]
    nbest_averaging_interval: int
    early_stopping_criterion: Sequence[str]
    best_model_criterion: Sequence[Sequence[str]]
    val_scheduler_criterion: Sequence[str]
    deepspeed_config: Union[Path, str]

class DeepSpeedTrainer(Trainer):

    @classmethod
    @typechecked
    def build_options(cls, args: argparse.Namespace) -> DeepSpeedTrainerOptions:
        return build_dataclass(DeepSpeedTrainerOptions, args)

    @staticmethod
    def resume(checkpoint: Union[str, Path]):
        raise NotImplementedError

    @classmethod
    @typechecked
    def run(
        cls,
        model: Union[AbsESPnetModel, DeepSpeedEngine],
        train_iter_factory: AbsIterFactory,
        valid_iter_factory: AbsIterFactory,
        plot_attention_iter_factory: Optional[AbsIterFactory],
        trainer_options: DeepSpeedTrainerOptions,
        distributed_option: DistributedOption,
        **kwargs,
    ) -> None:
        
        # (1) arguments needed in previous trainer but not this one. Delete them
        del kwargs

        # (2) initailize deepspeed
        assert distributed_option.distributed, "Distributed training not enabled."
        assert torch.distributed.is_initialized(), "Distributed backend not enabled"

        deepspeed_config = json.load(open(trainer_options.deepspeed_config))
        trainer_options.train_dtype = cls.setup_data_dtype(deepspeed_config)
        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=deepspeed_config,
        )

        # (3) Setup reporter, output_dir, dataloader etc.
        output_dir = Path(trainer_options.output_dir)
        reporter = Reporter()
        if trainer_options.log_interval is None:
            trainer_options.log_interval = 1000
        
        # (4) epoch loop
        start_epoch = reporter.get_epoch() + 1
        if start_epoch == trainer_options.max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )
        
        for iepoch in range(start_epoch, trainer_options.max_epoch + 1):
            set_all_random_seed(trainer_options.seed + iepoch)
            reporter.set_epoch(iepoch)

            with reporter.observe("train") as sub_reporter:
                cls.train_one_epoch(
                    model=model,
                    iterator=train_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                    distributed_option=distributed_option,
                )
    
    @classmethod
    @typechecked
    def train_one_epoch(
        cls,
        model,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        reporter: SubReporter,
        options: DeepSpeedTrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        model.train()
        iterator_stop = torch.tensor(0).cuda()

        for iiter, (utt_id, batch) in enumerate(iterator):
            assert isinstance(batch, dict), type(batch)

            # (0) ensure all ranks have not finished.
            torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
            if iterator_stop > 0:
                break

            # (1) forward
            batch["utt_id"] = utt_id
            batch = to_device(batch, "cuda", dtype=options.train_dtype)
            loss, stats, weight = model(**batch)

            # (2) all-reduce statistics and logging on model side
            stats = {k: v for k, v in stats.items() if v is not None}
            stats, weight = recursive_average(stats, weight, True)
            reporter.register(stats, weight)

            # (3) backward and logging on trainer side
            loss = loss / weight * torch.distributed.get_world_size()
            model.backward(loss)
            model.step()

            reporter.register(dict(
                grad_norm = model.get_global_grad_norm(),
                loss_scale=model.loss_scale(),
                learning_rate=model.get_lr()[0],
            ))

            reporter.next()
            if iiter % options.log_interval == 0:
                logging.info(reporter.log_message(-options.log_interval))

        else:
            iterator_stop.fill_(1)
            torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

    @classmethod
    @typechecked
    def setup_data_dtype(cls, deepspeed_config: Dict):
        if "bf16" in deepspeed_config:
            return torch.bfloat16
        
        elif "fp16" in deepspeed_config:
            return torch.float16
        
        elif "amp" in deepspeed_config:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16
            
        else:
            return torch.float