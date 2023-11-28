#!/usr/bin/env python3
import argparse
import logging
import sys

import humanfriendly
import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.samplers.build_batch_sampler import BATCH_TYPES
from espnet2.tasks.spk import SpeakerTask
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.distributed_utils import (
    DistributedOption,
    free_port,
    get_master_port,
    get_node_rank,
    get_num_nodes,
    resolve_distributed_mode,
)
from espnet2.train.reporter import Reporter, SubReporter
from espnet2.utils import config_argparse
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.types import (
    humanfriendly_parse_size_or_none,
    int_or_none,
    str2bool,
    str2triple_str,
    str_or_none,
)
from espnet.utils.cli_utils import get_commandline_args


def inference(args):
    assert check_argument_types()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if args.ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(args.seed)

    # 2. define train args
    spk_model, spk_train_args = SpeakerTask.build_model_from_file(
        args.spk_train_config, args.spk_model_file, device
    )

    # 3. Overwrite args with inference args
    args = vars(args)
    args["valid_data_path_and_name_and_type"] = args["data_path_and_name_and_type"]
    args["valid_shape_file"] = args["shape_file"]
    args["preprocessor_conf"] = {
        "target_duration": args["target_duration"],
        "num_eval": args["num_eval"],
        "noise_apply_prob": 0.0,
        "rir_apply_prob": 0.0,
    }

    merged_args = vars(spk_train_args)
    merged_args.update(args)
    args = argparse.Namespace(**merged_args)

    # 4. Build data-iterator
    resolve_distributed_mode(args)
    distributed_option = build_dataclass(DistributedOption, args)
    distributed_option.init_options()

    iterator = SpeakerTask.build_iter_factory(
        args=args,
        distributed_option=distributed_option,
        mode="valid",
    )
    loader = iterator.build_iter(0)

    trainer_options = SpeakerTask.trainer.build_options(args)
    reporter = Reporter()

    # 5. Run inference for EER and minDCF calculation
    with reporter.observe("valid") as sub_reporter:
        SpeakerTask.trainer.validate_one_epoch(
            model=spk_model,
            iterator=loader,
            reporter=sub_reporter,
            options=trainer_options,
            distributed_option=distributed_option,
        )
    if not distributed_option.distributed or distributed_option.dist_rank == 0:
        logging.info(reporter.log_message())


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="SPK Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    _batch_type_help = ""
    for key, value in BATCH_TYPES.items():
        _batch_type_help += f'"{key}":\n{value}\n'
    group.add_argument(
        "--batch_type",
        type=str,
        default="folded",
        choices=list(BATCH_TYPES),
        help=_batch_type_help,
    )
    group.add_argument(
        "--batch_bins",
        type=int,
        default=1000000,
        help="The number of batch bins. Used if batch_type='length' or 'numel'",
    )
    group.add_argument(
        "--valid_batch_bins",
        type=int_or_none,
        default=None,
        help="If not given, the value of --batch_bins is used",
    )
    group.add_argument(
        "--valid_batch_type",
        type=str_or_none,
        default=None,
        choices=list(BATCH_TYPES) + [None],
        help="If not given, the value of --batch_type is used",
    )
    group.add_argument(
        "--max_cache_size",
        type=humanfriendly.parse_size,
        default=0.0,
        help="The maximum cache size for data loader. e.g. 10MB, 20GB.",
    )
    group.add_argument(
        "--max_cache_fd",
        type=int,
        default=32,
        help="The maximum number of file descriptors to be kept "
        "as opened for ark files. "
        "This feature is only valid when data type is 'kaldi_ark'.",
    )
    group.add_argument(
        "--valid_max_cache_size",
        type=humanfriendly_parse_size_or_none,
        default=None,
        help="The maximum cache size for validation data loader. e.g. 10MB, 20GB. "
        "If None, the 5 percent size of --max_cache_size",
    )
    group.add_argument("--shape_file", type=str, action="append", default=[])
    group.add_argument(
        "--input_size",
        type=int_or_none,
        default=None,
        help="The number of input dimension of the feature",
    )
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)
    group.add_argument(
        "--spk2utt",
        type=str,
        default="",
        help="Directory of spk2utt file to be used in label mapping",
    )
    group.add_argument(
        "--train_dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type for training.",
    )
    group.add_argument(
        "--use_amp",
        type=str2bool,
        default=False,
        help="Enable Automatic Mixed Precision. This feature requires pytorch>=1.6",
    )
    group.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
        help="Gradient norm threshold to clip",
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
        "--grad_clip_type",
        type=float,
        default=2.0,
        help="The type of the used p-norm for gradient clip. Can be inf",
    )
    group.add_argument(
        "--grad_noise",
        type=str2bool,
        default=False,
        help="The flag to switch to use noise injection to "
        "gradients during training",
    )
    group.add_argument(
        "--resume",
        type=str2bool,
        default=False,
        help="Enable resuming if checkpoint is existing",
    )
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
    group.add_argument(
        "--drop_last_iter",
        type=str2bool,
        default=False,
        help="Exclude the minibatch with leftovers.",
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--spk_train_config",
        type=str,
        help="SPK training configuration",
    )
    group.add_argument(
        "--spk_model_file",
        type=str,
        help="SPK model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )

    group = parser.add_argument_group("distributed training related")
    group.add_argument(
        "--dist_backend",
        default="nccl",
        type=str,
        help="distributed backend",
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
        default=None,
        type=int_or_none,
        help="number of nodes for distributed training",
    )
    group.add_argument(
        "--dist_rank",
        type=int_or_none,
        default=None,
        help="node rank for distributed training",
    )
    group.add_argument(
        # Not starting with "dist_" for compatibility to launch.py
        "--local_rank",
        type=int_or_none,
        default=None,
        help="local rank for distributed training. This option is used if "
        "--multiprocessing_distributed=false",
    )
    group.add_argument(
        "--dist_master_addr",
        default=None,
        type=str_or_none,
        help="The master address for distributed training. "
        "This value is used when dist_init_method == 'env://'",
    )
    group.add_argument(
        "--dist_master_port",
        default=None,
        type=int_or_none,
        help="The master port for distributed training"
        "This value is used when dist_init_method == 'env://'",
    )
    group.add_argument(
        "--dist_launcher",
        default=None,
        type=str_or_none,
        choices=["slurm", "mpi", None],
        help="The launcher type for distributed training",
    )
    group.add_argument(
        "--multiprocessing_distributed",
        default=False,
        type=str2bool,
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    group.add_argument(
        "--unused_parameters",
        type=str2bool,
        default=False,
        help="Whether to use the find_unused_parameters in "
        "torch.nn.parallel.DistributedDataParallel ",
    )
    group.add_argument(
        "--sharded_ddp",
        default=False,
        type=str2bool,
        help="Enable sharded training provided by fairscale",
    )

    group = parser.add_argument_group("trainer initialization related")
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
        nargs="+",
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
        "--keep_nbest_models",
        type=int,
        nargs="+",
        default=[10],
        help="Remove previous snapshots excluding the n-best scored epochs",
    )
    group.add_argument(
        "--nbest_averaging_interval",
        type=int,
        default=0,
        help="The epoch interval to apply model averaging and save nbest models",
    )
    group.add_argument(
        "--use_matplotlib",
        type=str2bool,
        default=True,
        help="Enable matplotlib logging",
    )
    group.add_argument(
        "--use_tensorboard",
        type=str2bool,
        default=True,
        help="Enable tensorboard logging",
    )
    group.add_argument(
        "--create_graph_in_tensorboard",
        type=str2bool,
        default=False,
        help="Whether to create graph in tensorboard",
    )
    group.add_argument(
        "--use_wandb",
        type=str2bool,
        default=False,
        help="Enable wandb logging",
    )
    group.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Specify wandb project",
    )
    group.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="Specify wandb id",
    )
    group.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Specify wandb entity",
    )
    group.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Specify wandb run name",
    )
    group.add_argument(
        "--wandb_model_log_interval",
        type=int,
        default=-1,
        help="Set the model log period",
    )
    group.add_argument(
        "--detect_anomaly",
        type=str2bool,
        default=False,
        help="Set torch.autograd.set_detect_anomaly",
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
        default=True,
        help="Enable cudnn-deterministic mode",
    )

    group = parser.add_argument_group("The inference hyperparameter related")
    group.add_argument(
        "--valid_batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument(
        "--target_duration",
        type=float,
        default=3.0,
        help="Duration (in seconds) of samples in a minibatch",
    )
    group.add_argument(
        "--num_eval",
        type=int,
        default=10,
        help="Number of segments to make from one utterance in the inference phase",
    )
    group.add_argument("--fold_length", type=int, action="append", default=[])
    group.add_argument(
        "--use_preprocessor",
        type=str2bool,
        default=True,
        help="Apply preprocessing to data or not",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    inference(args)


if __name__ == "__main__":
    main()
