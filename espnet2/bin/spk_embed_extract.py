#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from glob import glob

import humanfriendly
import numpy as np
import torch
from torch.multiprocessing.spawn import ProcessContext
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
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import (
    humanfriendly_parse_size_or_none,
    int_or_none,
    str2bool,
    str2triple_str,
    str_or_none,
)
from espnet.utils.cli_utils import get_commandline_args


def extract_embed(args):
    distributed_option = build_dataclass(DistributedOption, args)
    distributed_option.init_options()

    if not distributed_option.distributed or distributed_option.dist_rank == 0:
        if not distributed_option.distributed:
            _rank = ""
        else:
            _rank = (
                f":{distributed_option.dist_rank}/"
                f"{distributed_option.dist_world_size}"
            )

        logging.basicConfig(
            level=args.log_level,
            format=f"[{os.uname()[1].split('.')[0]}{_rank}]"
            f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        # Suppress logging if RANK != 0
        logging.basicConfig(
            level="ERROR",
            format=f"[{os.uname()[1].split('.')[0]}"
            f":{distributed_option.dist_rank}/{distributed_option.dist_world_size}]"
            f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    # Invoking torch.distributed.init_process_group
    distributed_option.init_torch_distributed()

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
    # NOTE(jeeweon): Temporarily disable distributed to let loader include all trials
    org_distributed = distributed_option.distributed
    distributed_option.distributed = False
    iterator = SpeakerTask.build_iter_factory(
        args=args,
        distributed_option=distributed_option,
        mode="valid",
    )
    distributed_option.distributed = org_distributed
    loader = iterator.build_iter(0)
    bs = (
        args.valid_batch_size // args.ngpu
        if distributed_option.distributed
        else args.valid_batch_size
    )

    trainer_options = SpeakerTask.trainer.build_options(args)
    reporter = Reporter()

    # 5. Run inference for EER and minDCF calculation
    with reporter.observe("valid") as sub_reporter:
        SpeakerTask.trainer.extract_embed(
            model=spk_model,
            iterator=loader,
            reporter=sub_reporter,
            options=trainer_options,
            distributed_option=distributed_option,
            output_dir=args.output_dir,
            custom_bs=bs,
            average=args.average_embd,
        )

    if distributed_option.distributed:
        torch.distributed.barrier()
    if not distributed_option.distributed or distributed_option.dist_rank == 0:
        # Combine dictionaries into one
        npzs = glob(args.output_dir + "/embeddings*.npz")
        logging.info(f"{npzs}")
        embd_dic = {}
        for npz in npzs:
            tmp_dic = dict(np.load(npz))
            embd_dic.update(tmp_dic)
        set_name = args.data_path_and_name_and_type[0][0].split("/")[-2]
        np.savez(args.output_dir + f"/{set_name}_embeddings", **embd_dic)
        for npz in npzs:
            os.remove(npz)


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="speaker embedding extraction",
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
        "--allow_multi_rates",
        type=str2bool,
        default=False,
        help="Whether to allow audios to have different sampling rates",
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
    group.add_argument(
        "--num_cohort_spk",
        type=int,
        default=5994,
        help="The number of cohort speakers in score norm",
    )
    group.add_argument(
        "--num_utt_per_spk",
        type=int,
        default=10,
        help="The number of utterances per speaker in score norm",
    )
    group.add_argument(
        "--utt_select_sec",
        type=int,
        default=8,
        help="Minimum duration for including the utt in cohort set in score norm",
    )
    group.add_argument(
        "--average_spk",
        type=str2bool,
        default=False,
        help="whether to average cohort embeds per speaker in score norm",
    )
    group.add_argument(
        "--adaptive_cohort_size",
        type=int,
        default=400,
        help="top-k cohort size in score norm",
    )
    group.add_argument(
        "--qmf_dur_thresh",
        type=int,
        default=6,
        help="threshold of duration to be considered as long in qmf trainset",
    )
    group.add_argument(
        "--qmf_num_trial_per_condition",
        type=int,
        default=5000,
        help="number of trials per condition in qmf trainset",
    )
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)
    group.add_argument("--average_embd", type=str2bool, default=False)
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
        "--no_forward_run",
        type=str2bool,
        default=False,
        help="Just only iterating data loading without "
        "model forwarding and training",
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
    group.add_argument(
        "--use_lora",
        type=str2bool,
        default=False,
        help="Enable LoRA based finetuning, see (https://arxiv.org/abs/2106.09685) "
        "for large pre-trained foundation models, like Whisper",
    )
    group.add_argument(
        "--save_lora_only",
        type=str2bool,
        default=True,
        help="Only save LoRA parameters or save all model parameters",
    )
    group.add_argument(
        "--lora_conf",
        action=NestedDictAction,
        default=dict(),
        help="Configuration for LoRA based finetuning",
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

    # "distributed" is decided using the other command args
    resolve_distributed_mode(args)
    if not args.distributed or not args.multiprocessing_distributed:
        extract_embed(args)

    else:
        assert args.ngpu > 1, args.ngpu
        # Multi-processing distributed mode: e.g. 2node-4process-4GPU
        # |   Host1     |    Host2    |
        # |   Process1  |   Process2  |  <= Spawn processes
        # |Child1|Child2|Child1|Child2|
        # |GPU1  |GPU2  |GPU1  |GPU2  |

        # See also the following usage of --multiprocessing-distributed:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        num_nodes = get_num_nodes(args.dist_world_size, args.dist_launcher)
        if num_nodes == 1:
            args.dist_master_addr = "localhost"
            args.dist_rank = 0
            # Single node distributed training with multi-GPUs
            if (
                args.dist_init_method == "env://"
                and get_master_port(args.dist_master_port) is None
            ):
                # Get the unused port
                args.dist_master_port = free_port()

        # Assume that nodes use same number of GPUs each other
        args.dist_world_size = args.ngpu * num_nodes
        node_rank = get_node_rank(args.dist_rank, args.dist_launcher)

        # The following block is copied from:
        # https://github.com/pytorch/pytorch/blob/master/torch/multiprocessing/spawn.py
        error_queues = []
        processes = []
        mp = torch.multiprocessing.get_context("spawn")
        for i in range(args.ngpu):
            # Copy args
            local_args = argparse.Namespace(**vars(args))

            local_args.local_rank = i
            local_args.dist_rank = args.ngpu * node_rank + i
            local_args.ngpu = 1

            process = mp.Process(
                target=extract_embed,
                args=(local_args,),
                daemon=False,
            )
            process.start()
            processes.append(process)
            error_queues.append(mp.SimpleQueue())
        # Loop on join until it returns True or raises an exception.
        while not ProcessContext(processes, error_queues).join():
            pass


if __name__ == "__main__":
    main()
