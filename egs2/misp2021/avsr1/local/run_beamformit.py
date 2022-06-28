#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import argparse
import os


def beamformit_worker(
    beamformit_tool, config_file, source_dir, channel_scp, output_root
):
    f = open(channel_scp)
    for line in f:
        show_id = line.split(" ")[0]
        store_dir = os.path.join("/", *line.split(" ")[1].split("/")[:-1])
        print("*" * 50)
        print(store_dir)
        print(show_id)
        print("*" * 50)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir, exist_ok=True)
        cmd = "{} -s {} -c {} --config_file {} -source_dir {} --result_dir {}".format(
            beamformit_tool, show_id, channel_scp, config_file, source_dir, store_dir
        )
        os.system(cmd)
    f.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "beamformit_tool",
        type=str,
        default="./../se_misp/BeamformIt-master/BeamformIt",
        help="path of beamformit tool",
    )
    parser.add_argument(
        "config_file",
        type=str,
        default="./conf/all_conf.cfg",
        help="path of config file",
    )
    parser.add_argument(
        "source_dir",
        type=str,
        default="/yrfs2/cv1/hangchen2/data/MISP_121h_WPE_/",
        help="wpe data dir",
    )
    parser.add_argument(
        "channel_scp",
        type=str,
        default="exp/wpe_tmp/channels_misp",
        help="path of config file",
    )
    parser.add_argument(
        "output_root",
        type=str,
        default="/yrfs2/cv1/hangchen2/data/MISP_121h_WPE_/",
        help="beamformit data dir",
    )

    args = parser.parse_args()
    beamformit_worker(
        beamformit_tool=args.beamformit_tool,
        config_file=args.config_file,
        source_dir=args.source_dir,
        channel_scp=args.channel_scp,
        output_root=args.output_root,
    )
