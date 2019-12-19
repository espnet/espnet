from __future__ import annotations

import argparse
import logging
import random
import sys
from abc import ABC
from abc import abstractmethod
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import configargparse
import numpy as np
import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet.utils.cli_utils import get_commandline_args
from espnet2.optimizers.sgd import SGD
from espnet2.schedulers.abs_scheduler import AbsBatchScheduler
from espnet2.schedulers.abs_scheduler import AbsEpochScheduler
from espnet2.schedulers.noam_lr import NoamLR
from espnet2.torch_utils.load_pretrained_model import load_pretrained_model
from espnet2.train.abs_e2e import AbsE2E
from espnet2.train.batch_sampler import ConstantBatchSampler
from espnet2.train.batch_sampler import SubsetSampler
from espnet2.train.batch_sampler import build_batch_sampler
from espnet2.train.class_choices import ClassChoices
from espnet2.train.dataset import ESPnetDataset
from espnet2.train.reporter import Reporter
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump


_classes = dict(
    adam=torch.optim.Adam,
    sgd=SGD,
    adadelta=torch.optim.Adadelta,
    adagrad=torch.optim.Adagrad,
    adamax=torch.optim.Adamax,
    asgd=torch.optim.ASGD,
    lbfgs=torch.optim.LBFGS,
    rmsprop=torch.optim.RMSprop,
    rprop=torch.optim.Rprop,
)
if LooseVersion(torch.__version__) >= LooseVersion("1.2.0"):
    _classes["adamw"] = torch.optim.AdamW
optimizer_choices = ClassChoices(
    "optim", classes=_classes, type_check=torch.optim.Optimizer, default="adagrad"
)

batch_scheduler_choices = ClassChoices(
    "bscheduler",
    dict(
        ReduceLROnPlateau=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lambdalr=torch.optim.lr_scheduler.LambdaLR,
        steplr=torch.optim.lr_scheduler.StepLR,
        multisteplr=torch.optim.lr_scheduler.MultiStepLR,
        exponentiallr=torch.optim.lr_scheduler.ExponentialLR,
        CosineAnnealingLR=torch.optim.lr_scheduler.CosineAnnealingLR,
    ),
    type_check=AbsEpochScheduler,
    default=None,
)

_classes = dict()
if LooseVersion(torch.__version__) >= LooseVersion("1.1.0"):
    _classes["noamlr"] = NoamLR
if LooseVersion(torch.__version__) >= LooseVersion("1.3.0"):
    _classes.update(
        cycliclr=torch.optim.lr_scheduler.CyclicLR,
        onecyclelr=torch.optim.lr_scheduler.OneCycleLR,
        CosineAnnealingWarmRestarts=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    )
epoch_scheduler_choices = ClassChoices(
    "escheduler", classes=_classes, type_check=AbsBatchScheduler, default=None
)


class AbsTask(ABC):
    # Use @staticmethod, or @classmethod,
    # instead of instance method to avoid God classes

    class_choices_list: List[ClassChoices] = [
        # --optim and --optim_conf
        optimizer_choices,
        # --escheduler and --escheduler_conf
        epoch_scheduler_choices,
        # --bscheduler and --bscheduler_conf
        batch_scheduler_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer: Trainer

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @abstractmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        pass

    @classmethod
    @abstractmethod
    def get_task_config(cls) -> Dict[str, dict]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_collate_fn(
        cls, args: argparse.Namespace
    ) -> Callable[[Sequence[Dict[str, np.ndarray]]], Dict[str, torch.Tensor]]:
        """Return "collate_fn", which is a callable object and
        will be given to pytorch DataLoader.

        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(collate_fn=cls.build_collate_fn(args), ...)

        In many cases, you can use our common collate_fn:

        >>> from espnet2.train.collate_fn import common_collate_fn

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def required_data_names(cls, train: bool = True) -> Tuple[str, ...]:
        """Define the required names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as following,

        >>> from espnet2.train.abs_e2e import AbsE2E
        >>> class Model(AbsE2E):
        ...     def forward(self, input, output, opt=None):  pass

        then "required_data_names" should be as

        >>> required_data_names = ('input', 'output')
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def optional_data_names(cls, train: bool = True) -> Tuple[str, ...]:
        """Define the optional names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as following,

        >>> from espnet2.train.abs_e2e import AbsE2E
        >>> class Model(AbsE2E):
        ...     def forward(self, input, output, opt=None):  pass

        then "optional_data_names" should be as

        >>> optional_data_names = ('opt',)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_model(cls, args: argparse.Namespace) -> AbsE2E:
        raise NotImplementedError

    @classmethod
    def add_arguments(
        cls, parser: configargparse.ArgumentParser = None
    ) -> configargparse.ArgumentParser:
        assert check_argument_types()
        if parser is None:
            parser = configargparse.ArgumentParser(
                description="base parser",
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
            )

        # NOTE(kamo): Use '_' instead of '-' to avoid confusion.
        #  I think '-' looks really confusing if it's written in yaml.

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        #  to provide --print_config mode. Instead of it, do as
        parser.set_defaults(required=["output_dir"])

        group = parser.add_argument_group("Common configuration")

        group.add_argument("--config", is_config_file=True, help="config file path")
        group.add_argument(
            "--print_config",
            action="store_true",
            help="Print the config file and exit",
        )

        group.add_argument(
            "--log_level",
            type=lambda x: x.upper(),
            default="INFO",
            choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
            help="The verbose level of logging",
        )

        group.add_argument("--output_dir", type=str_or_none, default=None)
        group.add_argument(
            "--ngpu",
            type=int,
            default=0,
            help="The number of gpus. 0 indicates CPU mode",
        )
        group.add_argument("--seed", type=int, default=0, help="Random seed")
        group.add_argument(
            "--collect_stats",
            type=str2bool,
            default=False,
            help='Perform on "collect stats" mode',
        )

        group = parser.add_argument_group("Trainer related")
        group.add_argument(
            "--max_epoch",
            type=int,
            default=40,
            help="The maximum number epoch to train",
        )
        group.add_argument(
            "--train_dtype",
            default="float32",
            choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
            help="Data type for training. O0,O1,.. flags require apex. "
            "See https://nvidia.github.io/apex/amp.html#opt-levels",
        )
        group.add_argument(
            "--patience",
            type=int_or_none,
            default=None,
            help="Number of epochs to wait without improvement "
            "before stopping the training",
        )
        group.add_argument(
            "--val_scheduler_criterion",
            type=str,
            nargs=2,
            default=("eval", "loss"),
            help="The criterion used for the value given to the scheduler. "
            'Give a pair referring the phase, "train" or "eval",'
            'and the criterion name. The mode specifying "min" or "max" can '
            "be changed by --escheduler_conf",
        )
        group.add_argument(
            "--early_stopping_criterion",
            type=str,
            nargs=3,
            default=("eval", "loss", "min"),
            help="The criterion used for judging of early stopping. "
            'Give a pair referring the phase, "train" or "eval",'
            'the criterion name and the mode, "min" or "max", e.g. "acc,max".',
        )
        group.add_argument(
            "--best_model_criterion",
            type=str2triple_str,
            action="append",
            default=[
                ("train", "loss", "min"),
                ("eval", "loss", "min"),
                ("train", "acc", "max"),
                ("eval", "acc", "max"),
            ],
            help="The criterion used for judging of the best model. "
            'Give a pair referring the phase, "train" or "eval",'
            'the criterion name, and the mode, "min" or "max", e.g. "acc,max".',
        )

        group.add_argument(
            "--grad_clip",
            type=float,
            default=5.0,
            help="Gradient norm threshold to clip",
        )
        group.add_argument(
            "--grad_noise",
            type=str2bool,
            default=False,
            help="The flag to switch to use noise injection to "
            "gradients during training",
        )
        group.add_argument(
            "--accum_grad",
            type=int,
            default=1,
            help="The number of gradient accumulation",
        )
        group.add_argument(
            "--log_interval",
            type=int_or_none,
            default=None,
            help="Show the logs every the number iterations in each epochs at the "
            "training phase. If None is given, it is decided according the number "
            "of training samples automatically .",
        )
        group.add_argument(
            "--keep_n_best_snapshot",
            type=int,
            default=10,
            help="Remove previous snapshots excluding the n-best scored epochs",
        )
        group.add_argument(
            "--num_att_plot",
            type=int,
            default=3,
            help="The number images to plot the outputs from attention. "
            "This option makes sense only when attention-based model",
        )
        group.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="The number of workers used for DataLoader",
        )
        group.add_argument(
            "--no_forward_run",
            type=str2bool,
            default=False,
            help="Just only iterating data loading without "
            "model forwarding and training",
        )
        group.add_argument(
            "--no_backward_run",
            type=str2bool,
            default=False,
            help="Performs data loading and model forwarding without backward "
            "operations, optimizer updating, etc.",
        )

        group = parser.add_argument_group("Resuming or transfer learning related")

        def epoch_type(value: str) -> Optional[Union[str, int]]:
            if value == "latest":
                return value
            elif value.lower() in ("none", "null", "nil"):
                return None
            else:
                v = int(value)
                if v < 0:
                    raise TypeError("must be 0 or more integer")
                return v

        egroup = group.add_mutually_exclusive_group()
        egroup.add_argument(
            "--resume_epoch",
            type=epoch_type,
            default=None,
            help='The training starts from the specified epoch. "latest" indicates '
            "the latest-epoch file found in output_path. If None or 0 are specified, "
            "then training starts from the initial state",
        )
        egroup.add_argument("--resume_path", type=str_or_none, default=None)

        group.add_argument("--pretrain_path", type=str, default=[], nargs="+")
        group.add_argument("--pretrain_key", type=str_or_none, default=[], nargs="+")

        group = parser.add_argument_group("BatchSampler related")
        group.add_argument(
            "--batch_size",
            type=int,
            default=20,
            help="The mini-batch size used for training",
        )
        group.add_argument(
            "--eval_batch_size",
            type=int_or_none,
            default=None,
            help="If not given, the value of --batch_size is used",
        )

        _batch_type_choices = ("const", "seq", "bin", "frame")
        group.add_argument(
            "--batch_type", type=str, default="seq", choices=_batch_type_choices,
        )
        group.add_argument(
            "--eval_batch_type",
            type=str_or_none,
            default=None,
            choices=_batch_type_choices + (None,),
            help="If not given, the value of --batch_type is used",
        )

        group.add_argument("--train_shape_file", type=str, action="append", default=[])
        group.add_argument("--eval_shape_file", type=str, action="append", default=[])

        group.add_argument("--max_length", type=int, action="append", default=[])

        group.add_argument(
            "--sort_in_batch",
            type=str_or_none,
            default="descending",
            choices=["descending", "ascending", None],
            help="Sort the samples in each mini-batches by the sample "
            'lengths. To enable this, "shape_file" must have the length information.',
        )
        group.add_argument(
            "--sort_batch",
            type=str_or_none,
            default="descending",
            choices=["descending", "ascending"],
            help="Sort mini-batches by the sample lengths",
        )

        group = parser.add_argument_group("Dataset related")
        group.add_argument(
            "--train_data_path_and_name_and_type",
            type=str2triple_str,
            action="append",
            default=[],
        )
        group.add_argument(
            "--eval_data_path_and_name_and_type",
            type=str2triple_str,
            action="append",
            default=[],
        )
        group.add_argument(
            "--allow_variable_data_keys",
            type=str2bool,
            default=False,
            help="Allow the arbitrary keys for mini-batch with ignoring "
            "the task requirements",
        )

        group = parser.add_argument_group("Variable objects")
        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --optim and --optim_conf
            class_choices.add_arguments(group)

        cls.add_task_arguments(parser)
        assert check_return_type(parser)
        return parser

    @classmethod
    def exclude_opts(cls) -> Tuple[str, ...]:
        """The options not to be shown by --print_config"""
        return "required", "print_config", "config", "ngpu"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        # This method is used only for --print_config
        assert check_argument_types()
        parser = cls.add_arguments()
        args, _ = parser.parse_known_args()
        config = vars(args)
        # Excludes the options not to be shown
        for k in AbsTask.exclude_opts():
            config.pop(k)

        for class_choices in cls.class_choices_list:
            if getattr(args, class_choices.name) is not None:
                class_obj = class_choices.get_class(getattr(args, class_choices.name))
                conf = get_default_kwargs(class_obj)
                name = class_choices.name
                # Overwrite the default by the arguments,
                conf.update(config[f"{name}_conf"])
                # and set it again
                config[f"{name}_conf"] = conf

        task_config = cls.get_task_config()
        for k in task_config:
            if k not in config:
                raise RuntimeError(f"{k} is not in argparse options")
            assert isinstance(task_config[k], dict), (k, type(task_config[k]))
            if config[k] is not None:
                if not isinstance(config[k], dict):
                    raise RuntimeError(f"The default value of --{k} is not dict.")
                # The default kwargs is overwritten by command line args
                task_config[k].update(config[k])
                # Set it again to config
                config[k] = task_config[k]
        return config

    @classmethod
    def check_required_command_args(cls, args: argparse.Namespace):
        assert check_argument_types()
        for k in vars(args):
            if "-" in k:
                raise RuntimeError(
                    f'Use "_" instead of "-": parser.add_arguments("{k}")'
                )
        if len(args.pretrain_path) != len(args.pretrain_key):
            raise RuntimeError(
                "The number of --pretrain_path and --pretrain_key must be same"
            )

        required = ", ".join(
            f"--{a}" for a in args.required if getattr(args, a) is None
        )

        if len(required) != 0:
            parser = cls.add_arguments()
            parser.print_help(file=sys.stderr)
            p = Path(sys.argv[0]).name
            print(file=sys.stderr)
            print(
                f"{p}: error: the following arguments are required: " f"{required}",
                file=sys.stderr,
            )
            sys.exit(2)

    @classmethod
    def check_task_requirements(
        cls, dataset: ESPnetDataset, allow_variable_data_keys: bool, train: bool = True,
    ) -> None:
        """Check if the dataset satisfy the requirement of current Task"""
        assert check_argument_types()
        mes = (
            f"If you intend to use an additional input, modify "
            f'"{cls.__name__}.required_data_names()" or '
            f'"{cls.__name__}.optional_data_names()". '
            f"Otherwise you need to set --allow_variable_data_keys true "
        )

        for k in cls.required_data_names(train):
            if not dataset.has_name(k):
                raise RuntimeError(
                    f'"{cls.required_data_names(train)}" are required for'
                    f' {cls.__name__}. but "{dataset.names()}" are input.\n{mes}'
                )
        if not allow_variable_data_keys:
            task_keys = cls.required_data_names(train) + cls.optional_data_names(train)
            for k in dataset.names():
                if k not in task_keys:
                    raise RuntimeError(
                        f"The data-name must be one of {task_keys} "
                        f'for {cls.__name__}: "{k}" is not allowed.\n{mes}'
                    )

    @classmethod
    def print_config(cls, file=sys.stdout) -> None:
        assert check_argument_types()
        # Shows the config: e.g. python train.py asr --print_config
        config = cls.get_default_config()
        file.write(yaml_no_alias_safe_dump(config, indent=4, sort_keys=False))

    @classmethod
    def main(cls, args: argparse.Namespace = None, cmd: Sequence[str] = None) -> None:
        assert check_argument_types()
        print(get_commandline_args(), file=sys.stderr)
        if args is None:
            parser = cls.add_arguments()
            args = parser.parse_args(cmd)
        if args.print_config:
            cls.print_config()
            sys.exit(0)
        cls.check_required_command_args(args)
        logging.basicConfig(
            level=args.log_level,
            format="%(asctime)s (%(module)s:%(lineno)d) " "%(levelname)s: %(message)s",
        )

        # 1. Set random-seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)

        # 2. Build train-data-iterator
        if args.train_dtype in ("float32", "O0", "O1", "O2", "O3"):
            dtype = "float32"
        else:
            dtype = args.train_dtype
        train_dataset = ESPnetDataset(
            args.train_data_path_and_name_and_type,
            float_dtype=dtype,
            preprocess=cls.build_preprocess_fn(args, True),
        )
        cls.check_task_requirements(train_dataset, args.allow_variable_data_keys)
        train_batch_sampler = build_batch_sampler(
            type=args.batch_type,
            shape_files=args.train_shape_file,
            max_lengths=args.max_length,
            batch_size=args.batch_size,
            shuffle=True,
            sort_in_batch=args.sort_in_batch,
            sort_batch=args.sort_batch,
        )
        train_iter = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=cls.build_collate_fn(args),
            num_workers=args.num_workers,
        )

        # 3. Build eval-data-iterator
        eval_dataset = ESPnetDataset(
            args.eval_data_path_and_name_and_type,
            float_dtype=dtype,
            preprocess=cls.build_preprocess_fn(args, False),
        )
        cls.check_task_requirements(eval_dataset, args.allow_variable_data_keys)
        if args.eval_batch_type is None:
            args.eval_batch_type = args.batch_type
        if args.eval_batch_size is None:
            args.eval_batch_size = args.batch_size
        eval_batch_sampler = build_batch_sampler(
            type=args.eval_batch_type,
            shape_files=args.eval_shape_file,
            max_lengths=args.max_length,
            batch_size=args.eval_batch_size,
            shuffle=False,
            sort_in_batch=args.sort_in_batch,
            sort_batch=args.sort_batch,
        )
        eval_iter = DataLoader(
            dataset=eval_dataset,
            batch_sampler=eval_batch_sampler,
            collate_fn=cls.build_collate_fn(args),
            num_workers=args.num_workers,
        )

        # 4. Build a iterator used for attention plot
        if args.num_att_plot != 0:
            plot_attention_sampler = SubsetSampler(
                ConstantBatchSampler(
                    key_file=args.eval_shape_file[0], batch_size=1, shuffle=False,
                ),
                args.num_att_plot,
            )
            plot_attention_iter = DataLoader(
                dataset=eval_dataset,
                batch_sampler=plot_attention_sampler,
                collate_fn=cls.build_collate_fn(args),
                num_workers=args.num_workers,
            )
        else:
            plot_attention_iter = None

        # 5. Build model
        model = cls.build_model(args=args)
        if not isinstance(model, AbsE2E):
            raise RuntimeError(
                f"model must inherit {AbsE2E.__name__}, but got {type(model)}"
            )

        if args.train_dtype in ("float16", "float32", "float64"):
            dtype = getattr(torch, args.train_dtype)
        else:
            dtype = torch.float32
        model = model.to(dtype=dtype, device="cuda" if args.ngpu > 0 else "cpu")

        # 6. Build optimizer
        optimizer_class = optimizer_choices.get_class(args.optim)
        optimizer = optimizer_class(model.parameters(), **args.optim_conf)

        # 7. Build epoch_scheduler: invoked at every epochs
        # e.g. torch.optim.lr_scheduler.StepLR
        if args.escheduler is not None:
            epoch_scheduler_class = epoch_scheduler_choices.get_class(args.escheduler)
            epoch_scheduler = epoch_scheduler_class(optimizer, **args.escheduler_conf)
        else:
            epoch_scheduler = None

        # 8. Build batch_scheduler: invoked after every updating
        # e.g. torch.optim.lr_scheduler.CyclicLR
        if args.bscheduler is not None:
            batch_scheduler_class = batch_scheduler_choices.get_class(args.bscheduler)
            batch_scheduler = batch_scheduler_class(optimizer, **args.bscheduler_conf)
        else:
            batch_scheduler = None

        # 9. Dump "args" to config.yaml
        # NOTE(kamo): "args" should be saved after object-buildings are done
        #  because they are allowed to modify "args".
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.yaml").open("w") as f:
            logging.info(f'Saving the configuration in {output_dir / "config.yaml"}')
            yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

        logging.info(f"Model:\n{model}")
        logging.info(f"Train Dataset: {train_dataset}")
        logging.info(f"Train BatchSampler: {train_batch_sampler}")
        logging.info(f"Eval Dataset: {eval_dataset}")
        logging.info(f"Eval BatchSampler: {eval_batch_sampler}")
        logging.info(f"Optimizer:\n{optimizer}")
        logging.info(f"Epoch scheduler: {epoch_scheduler}")
        logging.info(f"Batch scheduler: {batch_scheduler}")

        # 10. Loads pre-trained model
        for p, k in zip(args.pretrain_path, args.pretrain_key):
            # if p is None -> apply the state to the whole model.
            # elif p is str e.g. "encoder" -> apply the state to model.encoder
            load_pretrained_model(
                model=model,
                pretrain_path=p,
                pretrain_key=k,
                map_location="cuda" if args.ngpu > 0 else "cpu",
            )

        # 11. Resume the state from the previous epoch
        # Either one of --resume_epoch or --resume_path can be used.
        reporter = Reporter()
        cls.trainer.resume(
            model=model,
            optimizer=optimizer,
            reporter=reporter,
            output_dir=output_dir,
            batch_scheduler=batch_scheduler,
            epoch_scheduler=epoch_scheduler,
            resume_epoch=args.resume_epoch,
            resume_path=args.resume_path,
            map_location="cuda" if args.ngpu > 0 else "cpu",
        )

        # 12. Run
        if args.collect_stats:
            # Perform on collect_stats mode. This mode has two roles
            # - Derive the length and dimension of all input data
            # - Accumulate feats, square values, and the length for whitening
            cls.trainer.collect_stats(
                model=model,
                train_iter=train_iter,
                eval_iter=eval_iter,
                output_dir=args.output_dir,
                ngpu=args.ngpu,
                log_interval=args.log_interval,
            )
        else:
            # Core design note:
            #   Don't give args to run() directly!!!
            #   Instead of it, define give "Options" object and build here.
            (
                train_options,
                eval_options,
                plot_attention_options,
            ) = cls.trainer.build_options(args)

            # Start training
            cls.trainer.run(
                model=model,
                optimizer=optimizer,
                train_iter=train_iter,
                eval_iter=eval_iter,
                plot_attention_iter=plot_attention_iter,
                reporter=reporter,
                output_dir=output_dir,
                batch_scheduler=batch_scheduler,
                epoch_scheduler=epoch_scheduler,
                max_epoch=args.max_epoch,
                train_dtype=args.train_dtype,
                patience=args.patience,
                keep_n_best_snapshot=args.keep_n_best_snapshot,
                early_stopping_criterion=args.early_stopping_criterion,
                best_model_criterion=args.best_model_criterion,
                val_scheduler_criterion=args.val_scheduler_criterion,
                train_options=train_options,
                eval_options=eval_options,
                plot_attention_options=plot_attention_options,
            )

            # Generated n-best averaged model
            cls.trainer.average_nbest_models(
                reporter=reporter,
                output_dir=output_dir,
                best_model_criterion=args.best_model_criterion,
                nbest=args.keep_n_best_snapshot,
            )
