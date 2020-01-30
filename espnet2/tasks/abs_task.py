from abc import ABC
from abc import abstractmethod
import argparse
import dataclasses
from distutils.version import LooseVersion
import logging
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import configargparse
import humanfriendly
import numpy as np
import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
from typeguard import check_argument_types
from typeguard import check_return_type
import yaml

from espnet.utils.cli_utils import get_commandline_args
from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.main_funcs.collect_stats import collect_stats
from espnet2.optimizers.sgd import SGD
from espnet2.schedulers.noam_lr import NoamLR
from espnet2.schedulers.warmup_lr import WarmupLR
from espnet2.torch_utils.load_pretrained_model import load_pretrained_model
from espnet2.torch_utils.pytorch_version import pytorch_cudnn_version
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_e2e import AbsE2E
from espnet2.train.batch_sampler import build_batch_sampler
from espnet2.train.batch_sampler import ConstantBatchSampler
from espnet2.train.class_choices import ClassChoices
from espnet2.train.dataset import ESPnetDataset
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.epoch_iter_factory import AbsIterFactory
from espnet2.train.epoch_iter_factory import EpochIterFactory
from espnet2.train.reporter import Reporter
from espnet2.train.trainer import Trainer
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import humanfriendly_parse_size_or_none
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
    scheduler_classes.update(
        noamlr=NoamLR, warmuplr=WarmupLR,
    )
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


@dataclasses.dataclass
class IteratorOption:
    train_dtype: str
    max_length: Sequence[int]
    num_workers: int
    sort_in_batch: str
    sort_batch: str
    distributed: bool
    seed: int
    allow_variable_data_keys: bool
    ngpu: int


class AbsTask(ABC):
    # Use @staticmethod, or @classmethod,
    # instead of instance method to avoid God classes

    # If you need more than one optimizers, change this value in inheritance
    num_optimizers: int = 1
    trainer = Trainer
    class_choices_list: List[ClassChoices] = []
    iterator_option = IteratorOption

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
        """Return "collate_fn", which is a callable object and given to DataLoader.

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
    def required_data_names(cls, inference: bool = False) -> Tuple[str, ...]:
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
    def optional_data_names(cls, inference: bool = False) -> Tuple[str, ...]:
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

        group = parser.add_argument_group("distributed training related")
        group.add_argument(
            "--distributed",
            type=str2bool,
            default=False,
            help="Enable distributed training",
        )
        group.add_argument(
            "--dist_backend", default="nccl", type=str, help="distributed backend",
        )
        group.add_argument(
            "--dist_init_method",
            type=str,
            default="env://",
            help='if init_method="env://", env values of "MASTER_PORT", "MASTER_ADDR", '
            '"WORLD_SIZE", and "RANK" are referred.',
        )
        group.add_argument(
            "--dist_world_size",
            default=-1,
            type=int,
            help="number of nodes for distributed training",
        )
        group.add_argument(
            "--dist_rank",
            type=int,
            default=-1,
            help="node rank for distributed training",
        )

        group = parser.add_argument_group("cudnn mode related")
        group.add_argument(
            "--cudnn_enabled",
            type=str2bool,
            default=torch.backends.cudnn.enabled,
            help="Enable CUDNN",
        )
        group.add_argument(
            "--cudnn_benchmark",
            type=str2bool,
            default=torch.backends.cudnn.benchmark,
            help="Enable cudnn-benchmark mode",
        )
        group.add_argument(
            "--cudnn_deterministic",
            type=str2bool,
            default=torch.backends.cudnn.deterministic,
            help="Enable cudnn-deterministic mode",
        )

        group = parser.add_argument_group("collect stats mode related")
        group.add_argument(
            "--collect_stats",
            type=str2bool,
            default=False,
            help='Perform on "collect stats" mode',
        )
        group.add_argument(
            "--write_collected_feats",
            type=str2bool,
            default=False,
            help='Write the output features from the model when "collect stats" mode',
        )

        group = parser.add_argument_group("Trainer related")
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
            default=("valid", "loss"),
            help="The criterion used for the value given to the lr scheduler. "
            'Give a pair referring the phase, "train" or "valid",'
            'and the criterion name. The mode specifying "min" or "max" can '
            "be changed by --scheduler_conf",
        )
        group.add_argument(
            "--early_stopping_criterion",
            type=str,
            nargs=3,
            default=("valid", "loss", "min"),
            help="The criterion used for judging of early stopping. "
            'Give a pair referring the phase, "train" or "valid",'
            'the criterion name and the mode, "min" or "max", e.g. "acc,max".',
        )
        group.add_argument(
            "--best_model_criterion",
            type=str2triple_str,
            action="append",
            default=[
                ("train", "loss", "min"),
                ("valid", "loss", "min"),
                ("train", "acc", "max"),
                ("valid", "acc", "max"),
            ],
            help="The criterion used for judging of the best model. "
            'Give a pair referring the phase, "train" or "valid",'
            'the criterion name, and the mode, "min" or "max", e.g. "acc,max".',
        )
        group.add_argument(
            "--keep_n_best_checkpoints",
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
        group.add_argument(
            "--resume",
            type=str2bool,
            default=False,
            help="Enable resuming if checkpoint is existing",
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

        group = parser.add_argument_group("Pretraining model related")
        group.add_argument("--pretrain_path", type=str, default=[], nargs="*")
        group.add_argument("--pretrain_key", type=str_or_none, default=[], nargs="*")

        group = parser.add_argument_group("BatchSampler related")
        group.add_argument(
            "--num_iters_per_epoch",
            type=int_or_none,
            default=None,
            help="Restrict the number of iterations for training per epoch",
        )
        group.add_argument(
            "--batch_size",
            type=int,
            default=20,
            help="The mini-batch size used for training",
        )
        group.add_argument(
            "--valid_batch_size",
            type=int_or_none,
            default=None,
            help="If not given, the value of --batch_size is used",
        )

        _batch_type_choices = ("const_no_sort", "const", "seq", "bin", "frame")
        group.add_argument(
            "--batch_type", type=str, default="seq", choices=_batch_type_choices,
        )
        group.add_argument(
            "--valid_batch_type",
            type=str_or_none,
            default=None,
            choices=_batch_type_choices + (None,),
            help="If not given, the value of --batch_type is used",
        )

        group.add_argument("--train_shape_file", type=str, action="append", default=[])
        group.add_argument("--valid_shape_file", type=str, action="append", default=[])
        group.add_argument("--max_length", type=int, action="append", default=[])
        group.add_argument(
            "--sort_in_batch",
            type=str,
            default="descending",
            choices=["descending", "ascending"],
            help="Sort the samples in each mini-batches by the sample "
            'lengths. To enable this, "shape_file" must have the length information.',
        )
        group.add_argument(
            "--sort_batch",
            type=str,
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
            "--valid_data_path_and_name_and_type",
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
        group.add_argument(
            "--max_cache_size",
            type=humanfriendly.parse_size,
            default=0.0,
            help="The maximum cache size for data loader. e.g. 10MB, 20GB.",
        )
        group.add_argument(
            "--valid_max_cache_size",
            type=humanfriendly_parse_size_or_none,
            default=None,
            help="The maximum cache size for validation data loader. e.g. 10MB, 20GB. "
            "If None, the 5% size of --max_cache_size",
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
                "build_optimizers() must be overridden if num_optimizers != 1"
            )

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
        cls,
        dataset: ESPnetDataset,
        allow_variable_data_keys: bool,
        inference: bool = False,
    ) -> None:
        """Check if the dataset satisfy the requirement of current Task"""
        assert check_argument_types()
        mes = (
            f"If you intend to use an additional input, modify "
            f'"{cls.__name__}.required_data_names()" or '
            f'"{cls.__name__}.optional_data_names()". '
            f"Otherwise you need to set --allow_variable_data_keys true "
        )

        for k in cls.required_data_names(inference):
            if not dataset.has_name(k):
                raise RuntimeError(
                    f'"{cls.required_data_names(inference)}" are required for'
                    f' {cls.__name__}. but "{dataset.names()}" are input.\n{mes}'
                )
        if not allow_variable_data_keys:
            task_keys = cls.required_data_names(inference) + cls.optional_data_names(
                inference
            )
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

        # 0. Init distributed process
        distributed_option = build_dataclass(DistributedOption, args)
        distributed_option.init()
        if not distributed_option.distributed or distributed_option.dist_rank == 0:
            logging.basicConfig(
                level=args.log_level,
                format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )
        else:
            # Suppress logging if RANK != 0
            logging.basicConfig(
                level="ERROR",
                format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )

        # 1. Set random-seed
        set_all_random_seed(args.seed)
        torch.backends.cudnn.enabled = args.cudnn_enabled
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cudnn.deterministic = args.cudnn_deterministic

        # 2. Build iterator factories
        iterator_option = build_dataclass(cls.iterator_option, args)

        train_iter_factory, _, _ = cls.build_iter_factory(
            iterator_option=iterator_option,
            data_path_and_name_and_type=args.train_data_path_and_name_and_type,
            shape_files=args.train_shape_file,
            batch_type=args.batch_type,
            batch_size=args.batch_size,
            train=True,
            preprocess_fn=cls.build_preprocess_fn(args, train=True),
            collate_fn=cls.build_collate_fn(args),
            num_iters_per_epoch=args.num_iters_per_epoch,
            max_cache_size=args.max_cache_size,
        )
        if args.valid_batch_type is None:
            args.valid_batch_type = args.batch_type
        if args.valid_batch_size is None:
            args.valid_batch_size = args.batch_size
        if args.valid_max_cache_size is None:
            # Cache 5% of maximum size for validation loader
            args.valid_max_cache_size = 0.05 * args.max_cache_size
        valid_iter_factory, _, _ = cls.build_iter_factory(
            iterator_option=iterator_option,
            data_path_and_name_and_type=args.valid_data_path_and_name_and_type,
            shape_files=args.valid_shape_file,
            batch_type=args.valid_batch_type,
            batch_size=args.valid_batch_size,
            train=False,
            preprocess_fn=cls.build_preprocess_fn(args, train=False),
            collate_fn=cls.build_collate_fn(args),
            num_iters_per_epoch=None,
            max_cache_size=args.valid_max_cache_size,
        )
        if args.num_att_plot != 0:
            plot_attention_iter_factory, _, _ = cls.build_iter_factory(
                iterator_option=iterator_option,
                data_path_and_name_and_type=args.valid_data_path_and_name_and_type,
                shape_files=[args.valid_shape_file[0]],
                batch_type="const_no_sort",
                batch_size=1,
                train=False,
                preprocess_fn=cls.build_preprocess_fn(args, train=False),
                collate_fn=cls.build_collate_fn(args),
                num_batches=args.num_att_plot,
                num_iters_per_epoch=None,
                # num_att_plot should be a few sample ~ 3, so cache all data.
                max_cache_size=np.inf if args.max_cache_size != 0.0 else 0.0,
            )
        else:
            plot_attention_iter_factory = None

        # 3. Build model
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

        # 4. Build optimizer
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

        # 5. Build schedulers
        schedulers = []
        for i, optim in enumerate(optimizers, 1):
            suf = "" if i == 1 else str(i)
            name = getattr(args, f"scheduler{suf}")
            conf = getattr(args, f"scheduler{suf}_conf")
            if name is not None:
                cls_ = scheduler_classes.get(name)
                if cls_ is None:
                    raise ValueError(
                        f"must be one of {list(scheduler_classes)}: {name}"
                    )
                scheduler = cls_(optim, **conf)
            else:
                scheduler = None

            schedulers.append(scheduler)

        logging.info(pytorch_cudnn_version())
        logging.info(f"Model:\n{model}")
        for i, (o, s) in enumerate(zip(optimizers, schedulers), 1):
            suf = "" if i == 1 else str(i)
            logging.info(f"Optimizer{suf}:\n{o}")
            logging.info(f"Scheduler{suf}: {s}")

        # 6. Dump "args" to config.yaml
        # NOTE(kamo): "args" should be saved after object-buildings are done
        #  because they are allowed to modify "args".
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.yaml").open("w") as f:
            logging.info(f'Saving the configuration in {output_dir / "config.yaml"}')
            yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

        # 7. Loads pre-trained model
        for p, k in zip(args.pretrain_path, args.pretrain_key):
            load_pretrained_model(
                model=model,
                # Directly specify the model path e.g. exp/train/loss.best.pt
                pretrain_path=p,
                # if pretrain_key is None -> model
                # elif pretrain_key is str e.g. "encoder" -> model.encoder
                pretrain_key=k,
                map_location="cuda" if args.ngpu > 0 else "cpu",
            )

        # 8. Resume the training state from the previous epoch
        reporter = Reporter()
        if args.resume and (output_dir / "checkpoint.pth").exists():
            states = torch.load(
                output_dir / "checkpoint.pth",
                map_location="cuda" if args.ngpu > 0 else "cpu",
            )
            model.load_state_dict(states["model"])
            reporter.load_state_dict(states["reporter"])
            for optimizer, state in zip(optimizers, states["optimizers"]):
                optimizer.load_state_dict(state)
            for scheduler, state in zip(schedulers, states["schedulers"]):
                if scheduler is not None:
                    scheduler.load_state_dict(state)

            logging.info(
                f"The training was resumed using {output_dir / 'checkpoint.pth'}"
            )

        # 9. Run
        if args.collect_stats:
            # Perform on collect_stats mode. This mode has two roles
            # - Derive the length and dimension of all input data
            # - Accumulate feats, square values, and the length for whitening
            collect_stats(
                model=model,
                train_iter=train_iter_factory.build_iter(1, shuffle=False),
                valid_iter=valid_iter_factory.build_iter(1, shuffle=False),
                output_dir=output_dir,
                ngpu=args.ngpu,
                log_interval=args.log_interval,
                write_collected_feats=args.write_collected_feats,
            )
        else:
            # Don't give args to run() directly!!!
            # Instead of it, define "Options" object and build here.
            trainer_options = cls.trainer.build_options(args)

            # Start training
            cls.trainer.run(
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                train_iter_factory=train_iter_factory,
                valid_iter_factory=valid_iter_factory,
                plot_attention_iter_factory=plot_attention_iter_factory,
                reporter=reporter,
                output_dir=output_dir,
                max_epoch=args.max_epoch,
                seed=args.seed,
                patience=args.patience,
                keep_n_best_checkpoints=args.keep_n_best_checkpoints,
                early_stopping_criterion=args.early_stopping_criterion,
                best_model_criterion=args.best_model_criterion,
                val_scheduler_criterion=args.val_scheduler_criterion,
                trainer_options=trainer_options,
                distributed_option=distributed_option,
            )

            # Generated n-best averaged model
            average_nbest_models(
                reporter=reporter,
                output_dir=output_dir,
                best_model_criterion=args.best_model_criterion,
                nbest=args.keep_n_best_checkpoints,
            )

    @classmethod
    def build_iter_factory(
        cls,
        iterator_option: IteratorOption,
        data_path_and_name_and_type,
        shape_files: Union[Tuple[str, ...], List[str]],
        batch_type: str,
        train: bool,
        preprocess_fn,
        batch_size: int,
        collate_fn,
        num_batches: int = None,
        num_iters_per_epoch: int = None,
        max_cache_size: float = 0,
    ) -> Tuple[AbsIterFactory, ESPnetDataset, List[Tuple[str, ...]]]:
        """Build a factory object of mini-batch iterator.

        This object is invoked at every epochs to build the iterator for each epoch
        as following:

        >>> iter_factory, _, _ = cls.build_iter_factory(...)
        >>> for epoch in range(1, max_epoch):
        ...     for keys, batch in iter_fatory.build_iter(epoch):
        ...         model(**batch)

        The mini-batches for each epochs are fully controlled by this class.
        Note that the random seed used for shuffling is decided as "seed + epoch" and
        the generated mini-batches can be reproduces even when resuming.

        Note that the definition of "epoch" doesn't always indicate
        to run out of the whole training corpus.
        "--num_iters_per_epoch" option restricts the number of iterations for each epoch
        and the rest of samples for the originally epoch are left for the next epoch.
        e.g. If The number of mini-batches equals to 4, the following two are same:

        - 1 epoch without "--num_iters_per_epoch"
        - 4 epoch with "--num_iters_per_epoch" == 4

        Note that this iterators highly assumes seq2seq training fashion,
        so the mini-batch includes the variable length sequences with padding.
        If you need to use custom iterators, e.g. LM training with truncated-BPTT uses
        fixed-length data cut off from concatenated sequences and in such case,
        your new task may need to override this method and "IteratorOption" class
        as following:

        >>> @dataclasses.dataclass
        ... class YourIteratorOption:
        ...     foo: int
        ...     bar: str
        >>> class YourTask(AbsTask):
        ...    iterator_option: YourIteratorOption
        ...
        ...    @classmethod
        ...    def build_iter_factory(cls, ...):


        """
        assert check_argument_types()
        if iterator_option.train_dtype in ("float32", "O0", "O1", "O2", "O3"):
            train_dtype = "float32"
        else:
            train_dtype = iterator_option.train_dtype

        dataset = ESPnetDataset(
            data_path_and_name_and_type,
            float_dtype=train_dtype,
            preprocess=preprocess_fn,
            max_cache_size=max_cache_size,
        )
        cls.check_task_requirements(dataset, iterator_option.allow_variable_data_keys)

        if train:
            shuffle = True
        else:
            num_iters_per_epoch = None
            shuffle = False

        batch_sampler = build_batch_sampler(
            type=batch_type,
            shape_files=shape_files,
            max_lengths=iterator_option.max_length,
            batch_size=batch_size,
            sort_in_batch=iterator_option.sort_in_batch,
            sort_batch=iterator_option.sort_batch,
        )

        batches = list(batch_sampler)
        if num_batches is not None:
            batches = batches[:num_batches]

        if iterator_option.distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            batches = [batch[rank::world_size] for batch in batches]

        return (
            EpochIterFactory(
                dataset=dataset,
                batches=batches,
                seed=iterator_option.seed,
                num_iters_per_epoch=num_iters_per_epoch,
                shuffle=shuffle,
                num_workers=iterator_option.num_workers,
                collate_fn=collate_fn,
                pin_memory=iterator_option.ngpu > 0,
            ),
            dataset,
            batches,
        )

    # ~~~~~~~~~ The methods below are mainly used for inference ~~~~~~~~~
    @classmethod
    def build_model_from_file(
        cls,
        config_file: Union[Path, str],
        model_file: Union[Path, str] = None,
        device: str = "cpu",
    ) -> Tuple[AbsE2E, argparse.Namespace]:
        """This method is used for inference or fine-tuning.

        Args:
            config_file: The yaml file saved when training.
            model_file: The model file saved when training.
            device:

        """
        assert check_argument_types()
        config_file = Path(config_file)

        with config_file.open("r") as f:
            args = yaml.safe_load(f)
        args = argparse.Namespace(**args)
        model = cls.build_model(args)
        if not isinstance(model, AbsE2E):
            raise RuntimeError(
                f"model must inherit {AbsE2E.__name__}, but got {type(model)}"
            )
        model.to(device)
        if model_file is not None:
            model.load_state_dict(torch.load(model_file, map_location=device))

        return model, args

    @classmethod
    def build_non_sorted_iterator(
        cls,
        data_path_and_name_and_type,
        batch_size: int = 1,
        dtype: str = "float32",
        key_file: str = None,
        num_workers: int = 1,
        pin_memory: bool = False,
        preprocess_fn=None,
        collate_fn=None,
        inference: bool = True,
        allow_variable_data_keys: bool = False,
    ) -> Tuple[DataLoader, ESPnetDataset, ConstantBatchSampler]:
        """Create mini-batch iterator w/o shuffling and sorting by sequence lengths.

        Note that unlike the iterator for training, the shape files are not required
        for this iterator because any sorting is not done here.

        """
        assert check_argument_types()
        if dtype in ("float32", "O0", "O1", "O2", "O3"):
            dtype = "float32"

        dataset = ESPnetDataset(
            data_path_and_name_and_type, float_dtype=dtype, preprocess=preprocess_fn,
        )
        cls.check_task_requirements(dataset, allow_variable_data_keys, inference)

        if key_file is None:
            key_file, _, _ = data_path_and_name_and_type[0]
        batch_sampler = ConstantBatchSampler(batch_size=batch_size, key_file=key_file)

        logging.info(f"Batch sampler: {batch_sampler}")
        logging.info(f"dataset:\n{dataset}")

        # For backward compatibility for pytorch DataLoader
        if collate_fn is not None:
            kwargs = dict(collate_fn=collate_fn)
        else:
            kwargs = {}

        return (
            DataLoader(
                dataset=dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **kwargs,
            ),
            dataset,
            batch_sampler,
        )
