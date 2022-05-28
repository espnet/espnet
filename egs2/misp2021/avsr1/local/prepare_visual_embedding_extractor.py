#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import argparse
import codecs
import os

from tqdm import tqdm


def text2lines(textpath, lines_content=None):
    """
    read lines from text or write lines to txt
    :param textpath: filepath of text
    :param lines_content: list of lines or None, None means read
    :return: processed lines content for read while None for write
    """
    if lines_content is None:
        with codecs.open(textpath, "r") as handle:
            lines_content = handle.readlines()
        processed_lines = [
            *map(lambda x: x[:-1] if x[-1] in ["\n"] else x, lines_content)
        ]
        return processed_lines
    else:
        processed_lines = [
            *map(lambda x: x if x[-1] in ["\n"] else "{}\n".format(x), lines_content)
        ]
        with codecs.open(textpath, "w") as handle:
            handle.write("".join(processed_lines))
        return None


def generate_extractor_shell(video_dir, nj, python_path="", used_gpus=[0, 1, 2, 3]):
    print("Avail gpus: {}".format(used_gpus))
    process2gpu = [used_gpus[i % len(used_gpus)] for i in range(nj)]
    print("Allocate gpu for {} processes: {}".format(nj, process2gpu))
    roi_npz_lines = text2lines(os.path.join(video_dir, "roi.scp"))
    embedding_scp_lines = []
    extractor_shells = [["set -e"] for _ in range(nj)]
    i = 0
    for line in tqdm(roi_npz_lines):
        key, roi_npz_path = line.split(" ")
        embedding_npz_path = os.path.join(
            video_dir, "visual_embedding", "{}.npz".format(key)
        )
        if os.path.exists(roi_npz_path):
            embedding_scp_lines.append(
                "{} {}".format(key, os.path.abspath(embedding_npz_path))
            )
            if not os.path.exists(embedding_npz_path):
                extractor_shells[i % nj].append(
                    "CUDA_VISIBLE_DEVICES={} {}python extractor/main.py \
                        --extract-feats \
                        --config-path extractor/configs/lrw_resnet18_mstcn.json \
                        --model-path extractor/models/lrw_resnet18_mstcn.pth.tar \
                        --mouth-patch-path {} --mouth-embedding-out-path {}".format(
                        process2gpu[(i % nj)],
                        python_path,
                        roi_npz_path,
                        embedding_npz_path,
                    )
                )
                i += 1
    for i, extractor_shell in enumerate(extractor_shells):
        text2lines(
            textpath=os.path.join(
                video_dir, "visual_embedding", "log", "extract.{}.sh".format(i + 1)
            ),
            lines_content=extractor_shells[i],
        )
    text2lines(
        textpath=os.path.join(video_dir, "embedding.scp"),
        lines_content=embedding_scp_lines,
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser("prepare_visual_embedding_extractor")
    parser.add_argument(
        "video_dir",
        type=str,
        default="/yrfs1/intern/hangchen2/experiment/EASE",
        help="video_dir",
    )
    parser.add_argument("-nj", type=int, default=15, help="number of process")
    parser.add_argument(
        "-g", type=int, nargs="+", default=[0, 1, 2, 3], help="number of process"
    )
    parser.add_argument("-p", type=str, default="", help="python_path")
    args = parser.parse_args()

    print("prepare visual embedding extractor")
    generate_extractor_shell(
        video_dir=args.video_dir, nj=args.nj, python_path=args.p, used_gpus=args.g
    )
    print(
        "Done: all {} shell for visual embedding extractor, {}".format(
            args.nj,
            os.path.join(args.video_dir, "visual_embedding", "log", "extract.*.sh"),
        )
    )
