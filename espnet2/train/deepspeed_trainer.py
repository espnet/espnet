"""DeepSpeed Trainer Module"""

import argparse
import base64
import dataclasses
import errno
import json
import logging
import time

import humanfriendly
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

from torch.distributed import ReduceOp
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.main_funcs.average_nbest_models import average_nbest_models
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
    use_wandb: bool
    max_epoch: int
    deepspeed_config: Union[Path, str]
    best_model_criterion: Sequence[Sequence[str]]
    keep_nbest_models: Union[int, List[int]]


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

        _, client_states = model.load_checkpoint(ckpt_path)

        reporter.load_state_dict(client_states["reporter"])

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

        if isinstance(trainer_options.keep_nbest_models, int):
            keep_nbest_models = [trainer_options.keep_nbest_models]
        else:
            if len(trainer_options.keep_nbest_models) == 0:
                logging.warning("No keep_nbest_models is given. Change to [1]")
                trainer_options.keep_nbest_models = [1]
            keep_nbest_models = trainer_options.keep_nbest_models

        # (1) arguments needed in previous trainer but not this one. Delete them
        del kwargs

        # (2) initailize deepspeed
        if deepspeed is None:
            raise ImportError("Cannot proceed as deepspeed is not installed")
        deepspeed_config = cls.setup_deepspeed_config(trainer_options.deepspeed_config)
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

        start_time = time.perf_counter()
        for iepoch in range(start_epoch, trainer_options.max_epoch + 1):
            set_all_random_seed(trainer_options.seed + iepoch)
            reporter.set_epoch(iepoch)
            if dist.get_rank() == 0:
                if iepoch != start_epoch:
                    logging.info(
                        "{}/{}epoch started. Estimated time to finish: {}".format(
                            iepoch,
                            trainer_options.max_epoch,
                            humanfriendly.format_timespan(
                                (time.perf_counter() - start_time)
                                / (iepoch - start_epoch)
                                * (trainer_options.max_epoch - iepoch + 1)
                            ),
                        )
                    )
                else:
                    logging.info(f"{iepoch}/{trainer_options.max_epoch}epoch started")

            # (5.1) train one epoch
            with reporter.observe("train") as sub_reporter:
                cls.train_one_epoch(
                    model=model,
                    iterator=train_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                )

            # (5.2) valid one epoch
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
                if trainer_options.use_wandb:
                    reporter.wandb_log()

        if dist.get_rank() == 0:
            average_nbest_models(
                reporter=reporter,
                output_dir=output_dir,
                best_model_criterion=trainer_options.best_model_criterion,
                nbest=keep_nbest_models,
                use_deepspeed=True,
            )

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
        iterator_stop = torch.tensor(0).cuda()

        log_interval = options.log_interval
        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100
        use_wandb = options.use_wandb

        for iiter, (utt_id, batch) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch, dict), type(batch)

            with reporter.measure_time("step_time"):
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
                    if use_wandb:
                        reporter.wandb_log()

        else:
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
    def setup_deepspeed_config(cls, deepspeed_config: Union[str, Path]):
        try:
            return json.load(open(deepspeed_config))
        except (FileNotFoundError, OSError) as e:
            if isinstance(e, OSError) and e.errno != errno.ENAMETOOLONG:
                # OSError needed because configs are too long so we get FileNameTooLongError
                raise e
            logging.info(
                f"Loading deepspeed config: {deepspeed_config} (a Base64-encoded string)"
            )
            decoded_json_str = base64.b64decode(deepspeed_config).decode("utf-8")
            return json.loads(decoded_json_str)
