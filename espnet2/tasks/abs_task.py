from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from dataclasses import is_dataclass
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
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
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.utils.cli_utils import get_commandline_args
from espnet2.optimizers.sgd import SGD
from espnet2.schedulers.abs_scheduler import AbsEpochStepScheduler
from espnet2.schedulers.abs_scheduler import AbsScheduler
from espnet2.schedulers.abs_scheduler import AbsValEpochStepScheduler
from espnet2.schedulers.noam_lr import NoamLR
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.forward_adaptor import ForwardAdaptor
from espnet2.train.abs_e2e import AbsE2E
from espnet2.train.batch_sampler import ConstantBatchSampler
from espnet2.train.batch_sampler import SubsetSampler
from espnet2.train.batch_sampler import build_batch_sampler
from espnet2.train.checkpoint import load_pretrained_model
from espnet2.train.checkpoint import resume
from espnet2.train.checkpoint import save_checkpoint
from espnet2.train.class_choices import ClassChoices
from espnet2.train.dataset import ESPnetDataset
from espnet2.train.reporter import Reporter
from espnet2.train.trainer import Trainer
from espnet2.utils.fileio import DatadirWriter
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump

optim_classes = dict(
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
    optim_classes["adamw"] = torch.optim.AdamW
scheduler_classes = dict(
    ReduceLROnPlateau=torch.optim.lr_scheduler.ReduceLROnPlateau,
    lambdalr=torch.optim.lr_scheduler.LambdaLR,
    steplr=torch.optim.lr_scheduler.StepLR,
    multisteplr=torch.optim.lr_scheduler.MultiStepLR,
    exponentiallr=torch.optim.lr_scheduler.ExponentialLR,
    CosineAnnealingLR=torch.optim.lr_scheduler.CosineAnnealingLR,
)
if LooseVersion(torch.__version__) >= LooseVersion("1.1.0"):
    scheduler_classes["noamlr"] = NoamLR
if LooseVersion(torch.__version__) >= LooseVersion("1.3.0"):
    CosineAnnealingWarmRestarts = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    scheduler_classes.update(
        cycliclr=torch.optim.lr_scheduler.CyclicLR,
        onecyclelr=torch.optim.lr_scheduler.OneCycleLR,
        CosineAnnealingWarmRestarts=CosineAnnealingWarmRestarts,
    )
# To lower keys
optim_classes = {k.lower(): v for k, v in optim_classes.items()}
scheduler_classes = {k.lower(): v for k, v in scheduler_classes.items()}


class AbsTask(ABC):
    # Use @staticmethod, or @classmethod,
    # instead of instance method to avoid God classes

    # If you need more than one optimizers, change this value in inheritance
    num_optimizers: int = 1
    trainer = Trainer
    class_choices_list: List[ClassChoices] = []

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @abstractmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        pass

    @classmethod
    @abstractmethod
    def build_collate_fn(
        cls, args: argparse.Namespace
    ) -> Callable[[Sequence[Dict[str, np.ndarray]]], Dict[str, torch.Tensor]]:
        """Return "collate_fn", which is a callable object and
        will be given to pytorch DataLoader.

        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(collate_fn=cls.build_collate_fn(args), ...)

        In many cases, you can use our common collate_fn.
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
    def get_parser(cls) -> configargparse.ArgumentParser:
        assert check_argument_types()
        parser = configargparse.ArgumentParser(
            description="base parser",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )

        # NOTE(kamo): Use '_' instead of '-' to avoid confusion.
        #  I think '-' looks really confusing if it's written in yaml.

        # NOTE(kamo): get_parser(..., required=True) can't be used
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
        group.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="The number of workers used for DataLoader",
        )
        group.add_argument(
            "--num_att_plot",
            type=int,
            default=3,
            help="The number images to plot the outputs from attention. "
            "This option makes sense only when attention-based model",
        )
        group.add_argument(
            "--train_dtype",
            default="float32",
            choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
            help="Data type for training. O0,O1,.. flags require apex. "
            "See https://nvidia.github.io/apex/amp.html#opt-levels",
        )
        group.add_argument(
            "--log_interval",
            type=int_or_none,
            default=None,
            help="Show the logs every the number iterations in each epochs at the "
            "training phase. If None is given, it is decided according the number "
            "of training samples automatically .",
        )

        group = parser.add_argument_group("train() related")
        group.add_argument(
            "--max_epoch",
            type=int,
            default=40,
            help="The maximum number epoch to train",
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
            help="The criterion used for the value given to the lr scheduler. "
            'Give a pair referring the phase, "train" or "eval",'
            'and the criterion name. The mode specifying "min" or "max" can '
            "be changed by --scheduler_conf",
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
            "--keep_n_best_snapshot",
            type=int,
            default=10,
            help="Remove previous snapshots excluding the n-best scored epochs",
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
            "--no_forward_run",
            type=str2bool,
            default=False,
            help="Just only iterating data loading without "
            "model forwarding and training",
        )

        group = parser.add_argument_group("Pretraining model related")
        group.add_argument("--pretrain_path", type=str, default=[], nargs="*")
        group.add_argument("--pretrain_key", type=str_or_none, default=[], nargs="*")

        group = parser.add_argument_group("Resuming related")

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

        group = parser.add_argument_group("Optimizer related")
        for i in range(1, cls.num_optimizers + 1):
            suf = "" if i == 1 else str(i)
            group.add_argument(
                f"--optim{suf}",
                type=lambda x: x.lower(),
                default="adadelta",
                choices=list(optim_classes),
                help=f"The optimizer type",
            )
            group.add_argument(
                f"--optim{suf}_conf",
                action=NestedDictAction,
                default=dict(),
                help=f"The keyword arguments for optimizer",
            )
            group.add_argument(
                f"--scheduler{suf}",
                type=lambda x: str_or_none(x.lower()),
                default=None,
                choices=list(scheduler_classes) + [None],
                help=f"The lr scheduler type",
            )
            group.add_argument(
                f"--scheduler{suf}_conf",
                action=NestedDictAction,
                default=dict(),
                help=f"The keyword arguments for lr scheduler",
            )

        cls.trainer.add_arguments(parser)
        cls.add_task_arguments(parser)

        assert check_return_type(parser)
        return parser

    @classmethod
    def build_optimizers(
        cls, args: argparse.Namespace, model: torch.nn.Module,
    ) -> List[torch.optim.Optimizer]:
        if cls.num_optimizers != 1:
            raise RuntimeError(
                "build_optimizers() must be overridden if num_optimizers != 1")

        optim_class = optim_classes.get(args.optim)
        if optim_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        optim = optim_class(model.parameters(), **args.optim_conf)
        optimizers = [optim]
        return optimizers

    @classmethod
    def exclude_opts(cls) -> Tuple[str, ...]:
        """The options not to be shown by --print_config"""
        return "required", "print_config", "config", "ngpu"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return the configuration as dict.

        This method is used print_config()
        """

        def get_class_type(name: str, classes: dict):
            _cls = classes.get(name)
            if _cls is None:
                raise ValueError(f"must be one of {list(classes)}: {name}")
            return _cls

        # This method is used only for --print_config
        assert check_argument_types()
        parser = cls.get_parser()
        args, _ = parser.parse_known_args()
        config = vars(args)
        # Excludes the options not to be shown
        for k in AbsTask.exclude_opts():
            config.pop(k)

        for i in range(1, cls.num_optimizers + 1):
            suf = "" if i == 1 else str(i)
            name = config[f"optim{suf}"]
            optim_class = get_class_type(name, optim_classes)
            conf = get_default_kwargs(optim_class)
            # Overwrite the default by the arguments,
            conf.update(config[f"optim{suf}_conf"])
            # and set it again
            config[f"optim{suf}_conf"] = conf

            name = config[f"scheduler{suf}"]
            if name is not None:
                scheduler_class = get_class_type(name, scheduler_classes)
                conf = get_default_kwargs(scheduler_class)
                # Overwrite the default by the arguments,
                conf.update(config[f"scheduler{suf}_conf"])
                # and set it again
                config[f"scheduler{suf}_conf"] = conf

        for class_choices in cls.class_choices_list:
            if getattr(args, class_choices.name) is not None:
                class_obj = class_choices.get_class(getattr(args, class_choices.name))
                conf = get_default_kwargs(class_obj)
                name = class_choices.name
                # Overwrite the default by the arguments,
                conf.update(config[f"{name}_conf"])
                # and set it again
                config[f"{name}_conf"] = conf
        return config

    @classmethod
    def check_required_command_args(cls, args: argparse.Namespace):
        assert check_argument_types()
        for k in vars(args):
            if "-" in k:
                raise RuntimeError(f'Use "_" instead of "-": parser.get_parser("{k}")')
        if len(args.pretrain_path) != len(args.pretrain_key):
            raise RuntimeError(
                "The number of --pretrain_path and --pretrain_key must be same"
            )

        required = ", ".join(
            f"--{a}" for a in args.required if getattr(args, a) is None
        )

        if len(required) != 0:
            parser = cls.get_parser()
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
        if cls.num_optimizers != cls.trainer.num_optimizers:
            raise RuntimeError(
                f"Task.num_optimizers != Task.trainer.num_optimizers: "
                f"{cls.num_optimizers} != {cls.trainer.num_optimizers}"
            )

        assert check_argument_types()
        print(get_commandline_args(), file=sys.stderr)
        if args is None:
            parser = cls.get_parser()
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

        logging.info(f"Model:\n{model}")
        logging.info(f"Train Dataset: {train_dataset}")
        logging.info(f"Train BatchSampler: {train_batch_sampler}")
        logging.info(f"Eval Dataset: {eval_dataset}")
        logging.info(f"Eval BatchSampler: {eval_batch_sampler}")

        # 6. Build optimizer
        optimizers = cls.build_optimizers(args, model=model)

        # For apex support
        use_apex = args.train_dtype in ("O0", "O1", "O2", "O3")
        if use_apex:
            try:
                from apex import amp
            except ImportError:
                logging.error(
                    f"You need to install apex. "
                    f"See https://github.com/NVIDIA/apex#linux"
                )
                raise
            model, optimizers = amp.initialize(
                model, optimizers, opt_level=args.train_dtype
            )

        # 7. Build schedulers
        schedulers = []
        for i, optim in enumerate(optimizers, 1):
            suf = "" if i == 1 else str(i)
            name = getattr(args, f"scheduler{suf}")
            conf = getattr(args, f"scheduler{suf}_conf")
            if name is not None:
                cls_ = scheduler_classes.get(name)
                if cls_ is None:
                    raise ValueError(f"must be one of {list(scheduler_classes)}: {name}")
                scheduler = cls_(optim, **conf)
            else:
                scheduler = None

            schedulers.append(scheduler)
        for i, (o, s) in enumerate(zip(optimizers, schedulers), 1):
            suf = "" if i == 1 else str(i)
            logging.info(f"Optimizer{suf}:\n{o}")
            logging.info(f"Scheduler{suf}: {s}")

        # 8. Dump "args" to config.yaml
        # NOTE(kamo): "args" should be saved after object-buildings are done
        #  because they are allowed to modify "args".
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.yaml").open("w") as f:
            logging.info(f'Saving the configuration in {output_dir / "config.yaml"}')
            yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

        # 9. Loads pre-trained model
        for p, k in zip(args.pretrain_path, args.pretrain_key):
            # if p is None -> apply the state to the whole model.
            # elif p is str e.g. "encoder" -> apply the state to model.encoder
            load_pretrained_model(
                pretrain_path=p,
                model=model,
                pretrain_key=k,
                map_location="cuda" if args.ngpu > 0 else "cpu",
            )

        # 10. Resume the state from the previous epoch
        # Either one of --resume_epoch or --resume_path can be used.
        reporter = Reporter()
        resume(
            model=model,
            reporter=reporter,
            output_dir=output_dir,
            optimizers=optimizers,
            schedulers=schedulers,
            resume_epoch=args.resume_epoch,
            resume_path=args.resume_path,
            map_location="cuda" if args.ngpu > 0 else "cpu",
        )

        # 11. Run
        if args.collect_stats:
            # Perform on collect_stats mode. This mode has two roles
            # - Derive the length and dimension of all input data
            # - Accumulate feats, square values, and the length for whitening
            cls.collect_stats(
                model=model,
                train_iter=train_iter,
                eval_iter=eval_iter,
                output_dir=args.output_dir,
                ngpu=args.ngpu,
                log_interval=args.log_interval,
            )
        else:
            # Don't give args to run() directly!!!
            # Instead of it, define give "Options" object and build here.
            trainer_options = cls.trainer.build_options(args)

            # Start training
            cls.run(
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                train_iter=train_iter,
                eval_iter=eval_iter,
                plot_attention_iter=plot_attention_iter,
                reporter=reporter,
                output_dir=output_dir,
                max_epoch=args.max_epoch,
                patience=args.patience,
                keep_n_best_snapshot=args.keep_n_best_snapshot,
                early_stopping_criterion=args.early_stopping_criterion,
                best_model_criterion=args.best_model_criterion,
                val_scheduler_criterion=args.val_scheduler_criterion,
                trainer_options=trainer_options,
            )

            # Generated n-best averaged model
            cls.average_nbest_models(
                reporter=reporter,
                output_dir=output_dir,
                best_model_criterion=args.best_model_criterion,
                nbest=args.keep_n_best_snapshot,
            )

    @classmethod
    @torch.no_grad()
    def collect_stats(
        cls,
        model: AbsE2E,
        train_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        eval_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        output_dir: Union[str, Path],
        ngpu: Optional[int],
        log_interval: Optional[int],
    ) -> None:
        """Perform on collect_stats mode.

        Running for deriving the shape information from data
        and gathering statistics.
        This method is used before executing train().

        """
        assert check_argument_types()
        output_dir = Path(output_dir)

        for itr, mode in zip([train_iter, eval_iter], ["train", "eval"]):
            if log_interval is None:
                log_interval = max(len(itr) // 20, 10)

            sum_dict = defaultdict(lambda: 0)
            sq_dict = defaultdict(lambda: 0)
            count_dict = defaultdict(lambda: 0)

            with DatadirWriter(output_dir / mode) as writer:
                for iiter, (keys, batch) in enumerate(itr, 1):
                    batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

                    # 1. Write shape file
                    for name in batch:
                        if name.endswith("_lengths"):
                            continue
                        for i, (k, data) in enumerate(zip(keys, batch[name])):
                            if f"{name}_lengths" in batch:
                                lg = int(batch[f"{name}_lengths"][i])
                                shape = ",".join(map(str, (lg,) + data.shape[1:]))
                            else:
                                shape = ",".join(map(str, data.shape))
                            writer[f"{name}_shape"][k] = shape

                    # 2. Extract feats and calc sum and square sum
                    if ngpu <= 1:
                        data = model.collect_feats(**batch)
                    else:
                        # Note that data_parallel can parallelize only "forward()"
                        data = data_parallel(
                            ForwardAdaptor(model, "collect_feats"),
                            (),
                            range(ngpu),
                            module_kwargs=batch,
                        )
                    for k, v in data.items():
                        if k.endswith("_lengths"):
                            continue
                        if f"{k}_lengths" in data:
                            # value: (Batch, Length, Dim, ...)
                            # -> Summation over batchxlength
                            ind = (0, 1)
                            count = v.size(0) * v.size(1)
                        else:
                            # value: (Batch, Dim, ...) -> Summation over batch
                            ind = 0
                            count = v.size(0)
                        v = v.cpu()
                        v.masked_fill_(make_pad_mask(data[f"{k}_lengths"], v, 1), 0.0)
                        sum_dict[k] += v.sum(ind).cpu().numpy()
                        sq_dict[k] += (v ** 2).sum(ind).cpu().numpy()
                        count_dict[k] += count

                    if iiter % log_interval == 0:
                        logging.info(f"Niter: {iiter}")

            for key in sum_dict:
                np.savez(
                    output_dir / mode / f"{key}_stats.npz",
                    count=count_dict[key],
                    sum=sum_dict[key],
                    sum_square=sq_dict[key],
                    )
            with (output_dir / mode / "shape_keys").open("w") as f:
                f.write(
                    "\n".join(filter(lambda x: not x.endswith("_lengths"), batch))
                    + "\n"
                )
            with (output_dir / mode / "stats_keys").open("w") as f:
                f.write("\n".join(sum_dict) + "\n")

    @classmethod
    def run(
        cls,
        model: AbsE2E,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        train_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        eval_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        plot_attention_iter,
        reporter: Reporter,
        output_dir: Union[str, Path],
        max_epoch: int,
        patience: Optional[int],
        keep_n_best_snapshot: int,
        early_stopping_criterion: Sequence[str],
        best_model_criterion: Sequence[Sequence[str]],
        val_scheduler_criterion: Sequence[str],
        trainer_options,
    ) -> None:
        """Perform training. This method performs the main process of training."""
        assert check_argument_types()
        # NOTE(kamo): Don't check the type more strict as far *_options
        assert is_dataclass(trainer_options), type(trainer_options)

        start_epoch = reporter.get_epoch() + 1
        if start_epoch == max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )

        best_epoch_dict = {}
        for iepoch in range(start_epoch, max_epoch + 1):
            logging.info(f"{iepoch}epoch started")

            reporter.set_epoch(iepoch)
            # 1. Train and eval for one-epoch
            with reporter.observe("train") as sub_reporter:
                all_steps_are_invalid = cls.trainer.train_one_epoch(
                    model=model,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    iterator=train_iter,
                    reporter=sub_reporter,
                    options=trainer_options,
                )
            with reporter.observe("eval") as sub_reporter:
                cls.trainer.eval_one_epoch(
                    model=model,
                    iterator=eval_iter,
                    reporter=sub_reporter,
                    options=trainer_options,
                )
            if plot_attention_iter is not None:
                with reporter.observe("att_plot") as sub_reporter:
                    cls.trainer.plot_attention(
                        model=model,
                        output_dir=output_dir / "att_ws" / f"{iepoch}epoch",
                        iterator=plot_attention_iter,
                        reporter=sub_reporter,
                        options=trainer_options,
                    )

            # 2. LR Scheduler step
            for scheduler in schedulers:
                if isinstance(scheduler, AbsValEpochStepScheduler):
                    _phase, _criterion = val_scheduler_criterion
                    if not reporter.has(_phase, _criterion):
                        raise RuntimeError(
                            f"{_phase}.{_criterion} is not found in stats: "
                            f"{reporter.get_all_keys()}"
                        )
                    val = reporter.get_value(_phase, _criterion)
                    scheduler.step(val)
                elif isinstance(scheduler, AbsEpochStepScheduler):
                    scheduler.step()

            # 3. Report the results
            reporter.logging()
            reporter.save_stats_plot(output_dir / "images")

            # 4. Save the snapshot
            save_checkpoint(
                save_path=output_dir / f"{iepoch}epoch",
                model=model,
                reporter=reporter,
                optimizers=optimizers,
                schedulers=schedulers,
            )

            # 5. Saves the best model
            _improved = []
            for _phase, k, _mode in best_model_criterion:
                if reporter.has(_phase, k):
                    best_epoch, _ = reporter.sort_epochs_and_values(_phase, k, _mode)[0]
                    best_epoch_dict[(_phase, k)] = best_epoch
                    # Creates sym links if it's the best result
                    if best_epoch == iepoch:
                        p = output_dir / f"{_phase}.{k}.best.pth"
                        if p.is_symlink() or p.exists():
                            p.unlink()
                        p.symlink_to(Path(f"{iepoch}epoch") / f"model.pth")
                        _improved.append(f"{_phase}.{k}")
            if len(_improved) == 0:
                logging.info(f"There are no improvements in this epoch")
            else:
                logging.info(
                    f"The best model has been updated: " + ", ".join(_improved)
                )

            # 6. Remove the snapshot excluding n-best and the current epoch
            _removed = []
            # nbests: List[List[Tuple[epoch, value]]]
            nbests = [
                reporter.sort_epochs_and_values(ph, k, m)[:keep_n_best_snapshot]
                for ph, k, m in best_model_criterion
                if reporter.has(ph, k)
            ]
            # nbests: Set[epoch]
            if len(nbests) != 0:
                nbests = set.union(*[set(i[0] for i in v) for v in nbests])
            else:
                nbests = set()
            for e in range(1, iepoch):
                p = output_dir / f"{e}epoch"
                if p.exists() and e not in nbests:
                    shutil.rmtree(p)
                    _removed.append(str(p))
            if len(_removed) != 0:
                logging.info(f"The snapshot was removed: " + ", ".join(_removed))

            # 7. If any updating haven't happen, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    f"The gradients at all steps are invalid in this epoch. "
                    f"Something seems wrong. This training was stopped at {iepoch}epoch"
                )
                break

            # 8. Check early stopping
            if patience is not None:
                _phase, _criterion, _mode = early_stopping_criterion
                if not reporter.has(_phase, _criterion):
                    raise RuntimeError(
                        f"{_phase}.{_criterion} is not found in stats: "
                        f"{reporter.get_all_keys()}"
                    )
                best_epoch, _ = reporter.sort_epochs_and_values(
                    _phase, _criterion, _mode
                )[0]
                if iepoch - best_epoch > patience:
                    logging.info(
                        f"[Early stopping] {_phase}.{_criterion} has not been "
                        f"improved {iepoch - best_epoch} epochs continuously. "
                        f"The training was stopped at {iepoch}epoch"
                    )
                    break

        else:
            logging.info(f"The training was finished at {max_epoch} epochs ")

    @classmethod
    @torch.no_grad()
    def average_nbest_models(
        cls,
        output_dir: Path,
        reporter: Reporter,
        best_model_criterion: Sequence[Sequence[str]],
        nbest: int,
    ) -> None:
        assert check_argument_types()
        # 1. Get nbests: List[Tuple[str, str, List[Tuple[epoch, value]]]]
        nbest_epochs = [
            (ph, k, reporter.sort_epochs_and_values(ph, k, m)[:nbest])
            for ph, k, m in best_model_criterion
            if reporter.has(ph, k)
        ]

        _loaded = {}
        for ph, cr, epoch_and_values in nbest_epochs:
            # Note that len(epoch_and_values) doesn't always equal to nbest.

            op = output_dir / f"{ph}.{cr}.ave_{len(epoch_and_values)}best.pth"
            logging.info(
                f"Averaging {len(epoch_and_values)}best models: "
                f'criterion="{ph}.{cr}": {op}'
            )

            avg = None
            # 2.a Averaging model
            for e, _ in epoch_and_values:
                if e not in _loaded:
                    _loaded[e] = torch.load(
                        output_dir / f"{e}epoch" / "model.pth", map_location="cpu",
                        )
                states = _loaded[e]

                if avg is None:
                    avg = states
                else:
                    # Accumulated
                    for k in avg:
                        avg[k] += states[k]
            for k in avg:
                avg[k] /= len(epoch_and_values)

            # 2.b Save the ave model and create a symlink
            torch.save(avg, op)
            sym_op = output_dir / f"{ph}.{cr}.ave.pth"
            if sym_op.is_symlink() or sym_op.exists():
                sym_op.unlink()
            sym_op.symlink_to(op.name)
