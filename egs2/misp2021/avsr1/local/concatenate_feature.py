#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import argparse
import codecs
import os

import kaldiio
import numpy as np
from tqdm import tqdm


def scp2array_dic(
    scp_path, array_dic=None, ark_path=None, compression_method=None, append=False
):
    """
    read array_dic from ark indexed by scp or
    write array_dic to ark while create scp to index
    :param scp_path: filepath of scp
    :param array_dic: dic of array
    :param ark_path: filepath of ark, default is scppath.replace('.scp', '.ark')
    :param compression_method: compression method, default=None,
                kAutomaticMethod=1, kSpeechFeature=2,
                kTwoByteAuto=3,kTwoByteSignedInteger=4, kOneByteAuto=5,
                kOneByteUnsignedInteger=6, kOneByteZeroOne=7
    :param append: if True, append, else write
    :return: dic of numpy array for read while None for write
    """
    if array_dic is None:
        array_dic = kaldiio.load_scp(scp_path)
        return array_dic
    else:
        if ark_path is None:
            ark_path = scp_path.replace(".scp", ".ark")
        else:
            pass
        kaldiio.save_ark(
            ark=ark_path,
            array_dict=array_dic,
            scp=scp_path,
            compression_method=compression_method,
            append=append,
        )
        return None


def main_concatenate(audio_dir, visual_dir, store_dir, ji=None, nj=None):
    audio_loader = scp2array_dic(
        scp_path=os.path.join(audio_dir, "feats.scp"),
        array_dic=None,
        ark_path=None,
        compression_method=None,
        append=False,
    )
    visual_npz_dic = {}
    with codecs.open(os.path.join(visual_dir, "embedding.scp"), "r") as handle:
        lines_content = handle.readlines()
    for line_content in [
        *map(lambda x: x[:-1] if x[-1] in ["\n"] else x, lines_content)
    ]:
        key, path = line_content.split(" ")
        visual_npz_dic[key] = path

    common_keys = [*(set(audio_loader.keys()) & set(visual_npz_dic.keys()))]
    store_scp = os.path.abspath(
        os.path.join(store_dir, "raw_av_embedding.{}.scp".format(ji))
    )
    store_ark = os.path.abspath(
        os.path.join(store_dir, "raw_av_embedding.{}.ark".format(ji))
    )

    for key_idx in tqdm(
        range(len(common_keys)), leave=True, desc="0" if ji is None else str(ji)
    ):
        if ji is None:
            processing_token = True
        else:
            if key_idx % nj == ji:
                processing_token = True
            else:
                processing_token = False
        if processing_token:
            key = common_keys[key_idx]
            audio_array = audio_loader[key]
            visual_array = np.load(visual_npz_dic[key])["data"][0]
            expend_visual_array = np.stack(
                [visual_array for _ in range(4)], axis=-1
            ).reshape(-1, visual_array.shape[-1])
            expend_visual_array = expend_visual_array[1:]
            expend_visual_array = expend_visual_array[: audio_array.shape[0]]
            audio_visual_array = np.concatenate(
                [audio_array, expend_visual_array], axis=-1
            )
            kaldiio.save_ark(
                ark=store_ark,
                array_dict={key: audio_visual_array},
                scp=store_scp,
                append=True,
            )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser("concatenate_feature")
    parser.add_argument(
        "audio_dir",
        type=str,
        default="data/train_far_sp_hire",
        help="data directory of audio",
    )
    parser.add_argument(
        "visual_dir",
        type=str,
        default="data/train_far_video",
        help="data directory of video",
    )
    parser.add_argument(
        "store_dir",
        type=str,
        default="data/test_far_av",
        help="store directory of av embedding",
    )
    parser.add_argument("--ji", type=int, default=0, help="index of process")
    parser.add_argument("--nj", type=int, default=15, help="number of process")

    args = parser.parse_args()

    nj = args.nj
    ji = args.ji if nj > 1 else 0

    main_concatenate(
        audio_dir=args.audio_dir,
        visual_dir=args.visual_dir,
        store_dir=args.store_dir,
        nj=nj,
        ji=ji,
    )
