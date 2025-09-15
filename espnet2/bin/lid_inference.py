#!/usr/bin/env python3
import argparse
import logging
import os
import random
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.multiprocessing.spawn import ProcessContext

from espnet2.samplers.build_batch_sampler import BATCH_TYPES
from espnet2.tasks.lid import LIDTask
from espnet2.torch_utils.model_summary import model_summary
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.distributed_utils import (
    DistributedOption,
    free_port,
    get_master_port,
    get_node_rank,
    get_num_nodes,
    resolve_distributed_mode,
)
from espnet2.train.reporter import Reporter
from espnet2.utils import config_argparse
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import (
    int_or_none,
    str2bool,
    str2triple_str,
    str_or_none,
)
from espnet.utils.cli_utils import get_commandline_args


def extract_embed_lid(args):
    """Perform inference for LID (Language Identification) tasks.

    This function loads a trained LID model, prepares the data iterator,
    extracts language ids per utterance, and extracts embeddings per language
    or utterance. It supports distributed inference, saving per-utterance or
    per-language embeddings, and can generate t-SNE plots for visualization.

    Args:
        args: The arguments containing model paths, data paths, inference
              options, and distributed settings. For example argument
              settings, refer to stage 6 and stage 8 in
              `egs2/TEMPLATE/lid1/lid.sh`.

    Note:
        - Supports both single-process and distributed inference.
        - Can save embeddings per utterance or per language.
        - Optionally generates t-SNE plots for visualization.
    """

    distributed_option = build_dataclass(DistributedOption, args)
    distributed_option.init_options()
    set_name = args.data_path_and_name_and_type[0][0].split("/")[-2]

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

    skip_extract_embd = False
    if os.path.exists(f"{args.output_dir}/lid_and_embd_extract.done"):
        skip_extract_embd = True

    if not skip_extract_embd:
        # 2. define train args
        lid_model, lid_train_args = LIDTask.build_model_from_file(
            args.lid_train_config, args.lid_model_file, device
        )
        logging.info(model_summary(lid_model))

        # 3. Overwrite args with inference args
        args = vars(args)
        args["valid_data_path_and_name_and_type"] = args["data_path_and_name_and_type"]
        args["valid_shape_file"] = args["shape_file"]
        args["preprocessor_conf"] = {
            "fix_duration": args["fix_duration"],
            "target_duration": args["target_duration"],
            "noise_apply_prob": 0.0,
            "rir_apply_prob": 0.0,
        }

        merged_args = vars(lid_train_args)
        merged_args.update(args)
        args = argparse.Namespace(**merged_args)

        # 4. Build data-iterator
        # NOTE(jeeweon): Temporarily disable distributed to
        # let loader include all trials
        org_distributed = distributed_option.distributed
        distributed_option.distributed = False

        if len(args.valid_data_path_and_name_and_type) == 1:
            # Only the speech is provided
            inference = True
        elif len(args.valid_data_path_and_name_and_type) == 2:
            # Both speech and lid labels are provided, for plotting tsne
            inference = False
        else:
            raise ValueError(
                "The number of valid_data_path_and_name_and_type must be 1 or 2, "
                f"but got {len(args.valid_data_path_and_name_and_type)}"
            )

        iterator = LIDTask.build_streaming_iterator(
            args.valid_data_path_and_name_and_type,
            dtype=args.dtype,
            batch_size=args.valid_batch_size,
            num_workers=args.num_workers,
            preprocess_fn=LIDTask.build_preprocess_fn(args, train=False),
            collate_fn=LIDTask.build_collate_fn(args, False),
            inference=inference,
        )
        distributed_option.distributed = org_distributed
        custom_bs = (
            args.valid_batch_size // args.ngpu
            if distributed_option.distributed
            else args.valid_batch_size
        )

        # 5. Create idx2lang dict, like {0: "eng", 1: "fra", ...}
        with open(args.lang2utt, "r") as f:
            lang2utt = f.readlines()
        lang_idx = 0
        lang2idx = {}
        for line in lang2utt:
            lang = line.strip().split()[0]
            lang2idx[lang] = lang_idx
            lang_idx += 1
        idx2lang = {v: k for k, v in lang2idx.items()}

        trainer_options = LIDTask.trainer.build_options(args)
        reporter = Reporter()

        # 6. Run inference
        lang_to_embds_dic = {
            lang_id: [] for lang_id in idx2lang.values()
        }  # {lang_id: [lang_embd for utt1, utt2, ...]}
        lang_counter_dic = {
            lang_id: 0 for lang_id in idx2lang.values()
        }  # {lang_id: num of utts}
        with reporter.observe("valid") as sub_reporter:
            LIDTask.trainer.extract_embed_lid(
                model=lid_model,
                iterator=iterator,
                reporter=sub_reporter,
                options=trainer_options,
                distributed_option=distributed_option,
                output_dir=args.output_dir,
                custom_bs=custom_bs,
                idx2lang=idx2lang,
                extract_embd=args.extract_embd,  # default False
                checkpoint_interval=args.checkpoint_interval,
                resume=args.resume,
                lang_to_embds_dic=lang_to_embds_dic,
                save_embd_per_utt=args.save_embd_per_utt,
                max_num_utt_per_lang=args.max_utt_per_lang_for_tsne,
                lang_counter_dic=lang_counter_dic,
            )

        # 7. Merge middle results from all processes and save final result
        # In distributed inference, each process saves its own results.
        # The main process merges all results to form the final output.
        if distributed_option.distributed:
            # lang_to_embds_dic is not shared between processes, so each
            # process must save its intermediate result for later merging
            # by the main process.
            np.savez(
                f"{args.output_dir}/"
                f"lang_to_embds_dic_rank{distributed_option.dist_rank}.npz",
                **lang_to_embds_dic,
            )
            torch.distributed.barrier()  # sync all processes
        if not distributed_option.distributed or distributed_option.dist_rank == 0:

            # Combine dictionaries into one
            if args.extract_embd and args.save_embd_per_utt:
                npzs = glob(f"{args.output_dir}/embeddings*.npz")
                logging.info(f"{npzs}")
                embd_dic = {}
                for npz in npzs:
                    tmp_dic = dict(np.load(npz))
                    embd_dic.update(tmp_dic)

                np.savez(f"{args.output_dir}/{set_name}_utt_embds", **embd_dic)
                for npz in npzs:
                    os.remove(npz)

            lid_files = glob(f"{args.output_dir}/lids*")
            lid_dic = {}
            for lid_file in lid_files:
                with open(lid_file, "r") as f:
                    for line in f:
                        utt_id, lid = line.strip().split()
                        lid_dic[utt_id] = lid

            with open(f"{args.output_dir}/{set_name}_lids", "w") as f:
                for utt_id, lid in lid_dic.items():
                    f.write(f"{utt_id} {lid}\n")
            for lid_file in lid_files:
                os.remove(lid_file)

            if distributed_option.distributed:
                # Merge lang_to_embds_dic of all processes
                merged_lang_to_embds_dic = {
                    lang_id: [] for lang_id in idx2lang.values()
                }
                for rank in range(distributed_option.dist_world_size):
                    npz_path = f"{args.output_dir}/lang_to_embds_dic_rank{rank}.npz"
                    if not os.path.exists(npz_path):
                        logging.warning(f"Missing {npz_path}, skipping.")
                        continue
                    npz = np.load(npz_path, allow_pickle=True)
                    for k in npz:
                        merged_lang_to_embds_dic[k].extend(list(npz[k]))

                np.savez(
                    f"{args.output_dir}/{set_name}_lang_to_embds",
                    **merged_lang_to_embds_dic,
                )
                # Remove temporary files
                for rank in range(distributed_option.dist_world_size):
                    npz_path = f"{args.output_dir}/lang_to_embds_dic_rank{rank}.npz"
                    if os.path.exists(npz_path):
                        os.remove(npz_path)
            else:
                np.savez(
                    f"{args.output_dir}/{set_name}_lang_to_embds",
                    **lang_to_embds_dic,
                )

            lang_to_avg_embd_dic = None
            logging.info(f"args.save_embd_per_utt: {args.save_embd_per_utt}")
            if args.extract_embd and args.save_embd_avg_lang:
                lang_to_avg_embd_dic = {}
                # Use the merged dictionary here
                if distributed_option.distributed:
                    use_dic = merged_lang_to_embds_dic
                else:
                    use_dic = lang_to_embds_dic
                for lang_id, embds in use_dic.items():
                    if len(embds) == 0:
                        continue
                    embds_array = np.stack(
                        embds, axis=0
                    )  # Stack list of ndarrays into a single ndarray
                    avg_embd = np.mean(
                        embds_array, axis=0
                    )  # Compute mean along the first axis
                    avg_embd = F.normalize(
                        torch.from_numpy(avg_embd), p=2, dim=0
                    ).numpy()
                    lang_to_avg_embd_dic[lang_id] = avg_embd
                np.savez(
                    f"{args.output_dir}/{set_name}_lang_to_avg_embd",
                    **lang_to_avg_embd_dic,
                )
    else:
        logging.info(
            f"Skipping language identification and embedding extraction for {set_name}."
        )

    if not distributed_option.distributed or distributed_option.dist_rank == 0:
        logging.info(f"args.save_tsne_plot: {args.save_tsne_plot}")

        if args.extract_embd and args.save_tsne_plot:
            lang_to_embds_path = f"{args.output_dir}/{set_name}_lang_to_embds.npz"
            if os.path.exists(lang_to_embds_path):
                use_dic = np.load(lang_to_embds_path)
                # Filter out languages with no embeddings
                use_dic = {
                    lang_id: embds
                    for lang_id, embds in use_dic.items()
                    if len(embds) > 0
                }
                gen_tsne_plot(
                    use_dic,
                    f"{args.output_dir}/tsne_plots",
                    args.seed,
                    perplexity=args.perplexity,
                    max_iter=args.max_iter,
                )
            else:
                logging.warning(
                    f"Missing {lang_to_embds_path}, "
                    "skipping plotting tsne_plot_lang_to_embds."
                )

            lang_to_avg_embd_path = f"{args.output_dir}/{set_name}_lang_to_avg_embd.npz"
            if os.path.exists(lang_to_avg_embd_path):
                lang_to_avg_embd_dic = np.load(lang_to_avg_embd_path)
                # Filter out languages with no embeddings
                lang_to_avg_embd_dic = {
                    lang_id: embds
                    for lang_id, embds in lang_to_avg_embd_dic.items()
                    if len(embds) > 0
                }
                gen_tsne_plot(
                    lang_to_avg_embd_dic,
                    f"{args.output_dir}/tsne_plots",
                    args.seed,
                    perplexity=args.perplexity,
                    max_iter=args.max_iter,
                )
            elif os.path.exists(lang_to_embds_path):
                use_dic = np.load(lang_to_embds_path)
                lang_to_avg_embd_dic = {}
                for lang_id, embds in use_dic.items():
                    if len(embds) == 0:
                        continue
                    embds_array = np.stack(
                        embds, axis=0
                    )  # Stack list of ndarrays into a single ndarray
                    avg_embd = np.mean(
                        embds_array, axis=0
                    )  # Compute mean along the first axis
                    lang_to_avg_embd_dic[lang_id] = avg_embd
                gen_tsne_plot(
                    lang_to_avg_embd_dic,
                    f"{args.output_dir}/tsne_plots",
                    args.seed,
                    perplexity=args.perplexity,
                    max_iter=args.max_iter,
                )
            else:
                logging.warning(
                    f"Missing {lang_to_avg_embd_path} and {lang_to_embds_path}, "
                    "skipping plotting tsne_plot_lang_to_avg_embd."
                )


def gen_tsne_plot(
    lang_to_embds_dic,
    output_dir,
    seed,
    perplexity=5,
    max_iter=1000,
):
    r"""Generate t-SNE plot for language embeddings with labels directly on the points.

    Args:
        lang_to_embds_dic (dict): Dictionary mapping language IDs (iso3 code)
                                  to embeddings.
        output_dir (str): Directory to save the t-SNE plot.
        seed (int): Random seed for reproducibility.
    """

    try:
        from adjustText import adjust_text

        has_adjust_text = True
    except ImportError:
        logging.warning("Please install adjustText: pip install adjustText")
        has_adjust_text = False
    try:
        import plotly.express as px

        has_px = True
    except ImportError:
        logging.warning("Please install plotly: pip install plotly")
        has_px = False
    try:
        import matplotlib.pyplot as plt

        has_plt = True
    except ImportError:
        logging.warning("Please install matplotlib: pip install matplotlib")
        has_plt = False
    try:
        import pandas as pd

        has_pd = True
    except ImportError:
        logging.warning("Please install pandas: pip install pandas")
        has_pd = False

    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare embeddings and labels
    embeddings = []
    labels = []

    logging.info("Preparing embeddings and labels...")
    total_num_embeddings = 0
    for lang_id, embds in sorted(lang_to_embds_dic.items()):
        if isinstance(embds, list):  # A list of embeddings for each language
            for embd in embds:
                embeddings.append(embd)
                labels.append(lang_id)
            plot_name = "lang_to_embds"
            total_num_embeddings += len(embds)
        elif isinstance(
            embds, np.ndarray
        ):  # A single embedding averaged across all utterances
            embeddings.append(embds)
            labels.append(lang_id)
            plot_name = "lang_to_avg_embd"
            total_num_embeddings += 1
        else:
            raise ValueError(f"Unsupported type for embeddings: {type(embds)}")

    if total_num_embeddings <= 1:
        logging.error(
            "Total number of embeddings should be greater than 1, "
            f"but got {total_num_embeddings}, "
            "skipping t-SNE plot."
        )
        return

    embeddings = np.array(embeddings)

    # Perform t-SNE with fixed random state
    logging.info("Performing t-SNE...")
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        max_iter=max_iter,
        init="pca",
        learning_rate="auto",
        n_jobs=1,
    )
    tsne_results = tsne.fit_transform(embeddings)

    if has_plt:
        # ========== Matplotlib Visualization ==========
        logging.info("Plotting t-SNE results...")
        plt.figure(figsize=(12, 10))

        fixed_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#aec7e8",
            "#ffbb78",
            "#98df8a",
            "#ff9896",
            "#c5b0d5",
            "#c49c94",
            "#f7b6d3",
            "#c7c7c7",
            "#dbdb8d",
            "#9edae5",
        ]

        if plot_name == "lang_to_embds":
            # Fixed color mapping
            unique_labels = sorted(list(set(labels)))

            if len(unique_labels) <= len(fixed_colors):
                color_dict = {
                    lang: fixed_colors[i] for i, lang in enumerate(unique_labels)
                }
            else:
                color_map = plt.get_cmap("tab20")
                color_dict = {
                    lang: color_map(i / len(unique_labels))
                    for i, lang in enumerate(unique_labels)
                }

            for lang_id in unique_labels:
                indices = [j for j, label in enumerate(labels) if label == lang_id]
                cluster_points = tsne_results[indices]
                # Calculate cluster center
                cluster_center = cluster_points.mean(axis=0)

                # Plot points for this language using consistent color
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=lang_id,
                    color=color_dict[lang_id],
                    alpha=0.7,
                    s=30,
                )

                # Add label at the cluster center
                plt.text(
                    cluster_center[0],
                    cluster_center[1],
                    lang_id,
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
                )
        elif plot_name == "lang_to_avg_embd":
            # Use scatter plot with text labels for each language
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7, s=30)
            texts = []
            for i, label in enumerate(labels):
                texts.append(
                    plt.text(
                        tsne_results[i, 0],
                        tsne_results[i, 1],
                        label,
                        fontsize=8,
                        alpha=0.8,
                    )
                )
            if has_adjust_text:
                adjust_text(
                    texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5)
                )
            else:
                logging.warning(
                    "Missing adjustText, "
                    "skipping adjusting text labels in t-SNE plot."
                )

        plt.title("t-SNE Visualization of Language Embeddings")

        # Save plot
        plot_file = f"{output_dir}/tsne_plot_{plot_name}.png"
        plt.savefig(plot_file, bbox_inches="tight", dpi=400)
        plt.close()
        logging.info(f"t-SNE plot saved to {plot_file}")
    else:
        logging.warning("Missing matplotlib.pyplot, skipping t-SNE plot.")

    df = None
    if has_pd:
        # Create DataFrame with consistent colors
        df = pd.DataFrame(
            {
                "x": tsne_results[:, 0],
                "y": tsne_results[:, 1],
                "label": labels,
            }
        )
        # =========== Save tsne results to CSV ===========
        df.to_csv(f"{output_dir}/tsne_results_{plot_name}.csv", index=False)
    else:
        logging.warning("Missing pandas, skipping saving t-SNE results to CSV.")

    if has_plt and has_px and df is not None:
        # ========== Plotly Interactive Visualization ==========
        unique_labels_plotly = sorted(df["label"].unique())
        if len(unique_labels_plotly) <= len(fixed_colors):
            color_discrete_map = {
                lang: fixed_colors[i] for i, lang in enumerate(unique_labels_plotly)
            }
        else:
            color_map = plt.get_cmap("tab20")
            color_discrete_map = {
                lang: color_map(i / len(unique_labels_plotly))
                for i, lang in enumerate(unique_labels_plotly)
            }

        # Plot with Plotly Express
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="label",
            hover_name="label",
            title="t-SNE Visualization of Language Embeddings (Interactive)",
            color_discrete_map=color_discrete_map,
        )

        # Save as HTML (interactive)
        plot_file = f"{output_dir}/tsne_plot_{plot_name}.html"
        fig.write_html(plot_file)
    else:
        missing_modules = []
        if not has_plt:
            missing_modules.append("matplotlib.pyplot")
        if df is None:
            missing_modules.append("pandas")
        if not has_px:
            missing_modules.append("plotly.express")
        logging.warning(
            f"Missing {missing_modules}, "
            "skipping visualizing t-SNE in interactive mode."
        )


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
    group.add_argument("--shape_file", type=str, action="append", default=[])
    group.add_argument(
        "--train_dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type for training.",
    )
    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--lid_train_config",
        type=str,
        help="LID training configuration",
    )
    group.add_argument(
        "--lid_model_file",
        type=str,
        help="LID model parameter file",
    )
    group.add_argument(
        "--extract_embd",
        type=str2bool,
        default=False,
        help="Determine whether to extract embedding or not",
    )
    group.add_argument(
        "--save_embd_per_utt",
        type=str2bool,
        default=False,
        help="Determine whether to save embedding for each utterance or not",
    )
    group.add_argument(
        "--save_embd_avg_lang",
        type=str2bool,
        default=False,
        help="Determine whether to save embedding for averaged across all utterances "
        "of a language or not",
    )
    group.add_argument(
        "--save_tsne_plot",
        type=str2bool,
        default=False,
        help="Determine whether to save tsne plot or not",
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
        "--init_param",
        default=None,
        type=str_or_none,
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
        "--fix_duration",
        type=str2bool,
        default=True,
        help="If True, fix the input duration to the target duration",
    )
    group.add_argument(
        "--target_duration",
        type=float,
        default=3.0,
        help="Duration (in seconds) of samples in a minibatch",
    )
    group.add_argument("--fold_length", type=int, action="append", default=[])
    group.add_argument(
        "--use_preprocessor",
        type=str2bool,
        default=True,
        help="Apply preprocessing to data or not",
    )
    group.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="Save checkpoint every N utterances for resume functionality",
    )
    group.add_argument(
        "--resume",
        type=str2bool,
        default=True,
        help="If True, avoid repeating existing inference results",
    )
    group.add_argument(
        "--max_utt_per_lang_for_tsne",
        type=int,
        default=None,
        help="The maximum number of utterances per language for t-SNE",
    )
    group.add_argument(
        "--perplexity",
        type=int,
        default=5,
        help="The perplexity for t-SNE",
    )
    group.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="The maximum number of iterations for t-SNE",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)

    # "distributed" is decided using the other command args
    resolve_distributed_mode(args)
    if not args.distributed or not args.multiprocessing_distributed:
        extract_embed_lid(args)

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
                target=extract_embed_lid,
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
