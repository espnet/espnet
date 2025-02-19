""" DeepSpeed Trainer Module """

import argparse
import dataclasses
import json
import logging

import torch
import torch.distributed as dist

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except ImportError:
    logging.warning("deepspeed is not installed")
    deepspeed = None
    DeepSpeedEngine = None

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import deepspeed
import torch
import torch.distributed as dist
from deepspeed import DeepSpeedEngine
from torch.distributed import ReduceOp
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.reporter import Reporter, SubReporter
from espnet2.train.trainer import Trainer
from espnet2.utils.build_dataclass import build_dataclass


@dataclasses.dataclass
class DeepSpeedTrainerOptions:
    resume: bool
    seed: int
    train_dtype: Union[str, torch.dtype]
    log_interval: Optional[int]
    output_dir: Union[Path, str]
    max_epoch: int
    torch_reseved_memory_gb: float
    deepspeed_step_sync: bool
    deepspeed_config: Union[Path, str]


class DeepSpeedTrainer(Trainer):

    @classmethod
    @typechecked
    def build_options(cls, args: argparse.Namespace) -> DeepSpeedTrainerOptions:
        return build_dataclass(DeepSpeedTrainerOptions, args)

    @staticmethod
    @typechecked
    def resume(
        model: DeepSpeedEngine,
        reporter: Reporter,
        output_dir: Path,
    ):
        ckpts = [
            item
            for item in output_dir.iterdir()
            if item.is_dir() and item.name.startswith("checkpoint_")
        ]

        if len(ckpts) == 0:
            logging.info("Try to resume but find no checkpoint")
            return

        ckpt_num = max([int(item.name.split("_")[-1]) for item in ckpts])
        ckpt_path = output_dir / f"checkpoint_{ckpt_num}"
        logging.info(f"Resume training from {ckpt_path}")

        _, clinet_states = model.load_checkpoint(ckpt_path)

        reporter.load_state_dict(clinet_states["reporter"])

    @classmethod
    @typechecked
    def run(
        cls,
        model: Union[AbsESPnetModel, DeepSpeedEngine],
        train_iter_factory: AbsIterFactory,
        valid_iter_factory: AbsIterFactory,
        trainer_options: DeepSpeedTrainerOptions,
        **kwargs,
    ) -> None:

        # (1) arguments needed in previous trainer but not this one. Delete them
        del kwargs

        # (2) initailize deepspeed
        if deepspeed is None:
            raise ImportError("Cannot proceed as deepspeed is not installed")
        deepspeed_config = json.load(open(trainer_options.deepspeed_config))
        trainer_options.train_dtype = cls.setup_data_dtype(deepspeed_config)
        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=deepspeed_config,
        )

        # (3) setup reporter, output_dir, dataloader etc.
        output_dir = Path(trainer_options.output_dir)
        reporter = Reporter()

        # (4) resume
        if trainer_options.resume:
            cls.resume(
                model=model,
                reporter=reporter,
                output_dir=output_dir,
            )

        # (5) loop on epochs
        start_epoch = reporter.get_epoch() + 1
        if start_epoch == trainer_options.max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )

        for iepoch in range(start_epoch, trainer_options.max_epoch + 1):
            set_all_random_seed(trainer_options.seed + iepoch)
            reporter.set_epoch(iepoch)

            # (5.1) train one epoch
            with reporter.observe("train") as sub_reporter:
                cls.train_one_epoch(
                    model=model,
                    iterator=train_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                )

            # (5.2) valid one epoch
            logging.info("Start Evaluation ...")
            with reporter.observe("valid") as sub_reporter:
                cls.valid_one_epoch(
                    model=model,
                    iterator=valid_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                )

            # (5.3) save checkpoint
            checkpoint_path = output_dir / f"checkpoint_{iepoch}"
            model.save_checkpoint(
                checkpoint_path,
                tag=f"{iepoch}",
                client_state={"reporter": reporter.state_dict()},
            )

            # (5.4) reporter
            if dist.get_rank() == 0:
                logging.info(reporter.log_message())
                reporter.matplotlib_plot(output_dir / "images")

    @classmethod
    @typechecked
    def train_one_epoch(
        cls,
        model,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        reporter: SubReporter,
        options: DeepSpeedTrainerOptions,
    ) -> None:
        model.train()
        torch.cuda.empty_cache()
        iterator_stop = torch.tensor(0).cuda()

        log_interval = options.log_interval
        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        if not options.deepspeed_step_sync:
            cls.check_iterator_length(iterator)

        cls.reserve_runtime_memory(size_gb=options.torch_reseved_memory_gb)

        for iiter, (utt_id, batch) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch, dict), type(batch)

            with reporter.measure_time("step_time"):
                # (0) ensure all ranks have not finished.
                if options.deepspeed_step_sync:
                    dist.all_reduce(iterator_stop, ReduceOp.SUM)
                    if iterator_stop > 0:
                        break

                # (1) forward
                batch["utt_id"] = utt_id
                batch = to_device(batch, "cuda", dtype=options.train_dtype)
                loss, stats, weight = model(**batch)

                # (2) all-reduce statistics and logging on model side
                stats = {k: v for k, v in stats.items() if v is not None}
                stats, weight = recursive_average(
                    stats, weight, options.deepspeed_step_sync
                )
                reporter.register(stats, weight)

                # (3) backward and logging on trainer side
                loss = loss / weight * dist.get_world_size()
                model.backward(loss)
                model.step()

                reporter.register(
                    dict(
                        grad_norm=model.get_global_grad_norm(),
                        loss_scale=model.loss_scale(),
                        learning_rate=model.get_lr()[0],
                    )
                )

                reporter.next()
                if iiter % log_interval == 0:
                    logging.info(reporter.log_message(-log_interval))
                    logging.info(
                        f"Pytorch memory reservation: "
                        f"{torch.cuda.memory_reserved() // 1024 ** 3} GB"
                    )

        else:
            if options.deepspeed_step_sync:
                iterator_stop.fill_(1)
                dist.all_reduce(iterator_stop, ReduceOp.SUM)

    @classmethod
    @typechecked
    @torch.no_grad()
    def valid_one_epoch(
        cls,
        model,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        reporter: SubReporter,
        options: DeepSpeedTrainerOptions,
    ) -> None:
        model.eval()
        torch.cuda.empty_cache()
        iterator_stop = torch.tensor(0).cuda()

        for iiter, (utt_id, batch) in enumerate(iterator):
            assert isinstance(batch, dict), type(batch)

            # (0) ensure all ranks have not finished.
            dist.all_reduce(iterator_stop, ReduceOp.SUM)
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
            reporter.next()

        else:
            iterator_stop.fill_(1)
            dist.all_reduce(iterator_stop, ReduceOp.SUM)

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

    @classmethod
    @typechecked
    def check_iterator_length(cls, iter: Iterable):
        this_length = torch.Tensor([len(iter)]).long().cuda()
        length_list = [
            torch.Tensor([0]).long().cuda() for _ in range(dist.get_world_size())
        ]
        dist.all_gather(length_list, this_length)
        length_list = torch.cat(length_list)
        assert torch.all(
            length_list.eq(this_length)
        ), f"Iterator lengths are different across all ranks: {length_list}"
    
    @classmethod
    def reserve_runtime_memory(cls, size_gb: float):
        if size_gb <= 0:
            return

        torch.cuda.empty_cache()
        logging.info(
            f"Before reserving memory: {torch.cuda.memory_reserved() // 1024 ** 3} GB"
        )
        
        tensors = []
        while torch.cuda.memory_reserved() // 1024 ** 3 < size_gb:
            try:
                tensor = torch.empty(1024 ** 3 // 2, dtype=torch.float16, device="cuda")
                tensors.append(tensor)
            except torch.cuda.OutOfMemoryError:
                logging.info("Cannot further reserve memory due to OOM")
                break
        del tensors

        logging.info(
            f"after reserving memory: {torch.cuda.memory_reserved() // 1024 ** 3} GB"
        )
