# Copyright 2020 The ESPnet Authors.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Speech enhancement."""

from itertools import zip_longest
import json
import logging
import os

import numpy as np
import torch

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import plot_spectrogram
from espnet.asr.asr_utils import torch_load
from espnet.nets.asr_interface import ASRInterface
from espnet.transform.spectrogram import IStft
from espnet.transform.transformation import Transformation
from espnet.utils.cli_writers import file_writer_helper
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets


def enhance(args):
    """Dump enhanced speech and mask.

    Args:
        args (namespace): The program arguments.
    """
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # TODO(ruizhili): implement enhance for multi-encoder model
    assert args.num_encs == 1, "number of encoder should be 1 ({} is given)".format(
        args.num_encs
    )

    # load trained model parameters
    logging.info("reading model parameters from " + args.model)
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    assert isinstance(model, ASRInterface)
    torch_load(args.model, model)
    model.recog_args = args

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=None,  # Apply pre_process in outer func
    )
    if args.batchsize == 0:
        args.batchsize = 1

    # Creates writers for outputs from the network
    if args.enh_wspecifier is not None:
        enh_writer = file_writer_helper(args.enh_wspecifier, filetype=args.enh_filetype)
    else:
        enh_writer = None

    # Creates a Transformation instance
    preprocess_conf = (
        train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf
    )
    if preprocess_conf is not None:
        logging.info(f"Use preprocessing: {preprocess_conf}")
        transform = Transformation(preprocess_conf)
    else:
        transform = None

    # Creates a IStft instance
    istft = None
    frame_shift = args.istft_n_shift  # Used for plot the spectrogram
    if args.apply_istft:
        if preprocess_conf is not None:
            # Read the conffile and find stft setting
            with open(preprocess_conf) as f:
                # Json format: e.g.
                #    {"process": [{"type": "stft",
                #                  "win_length": 400,
                #                  "n_fft": 512, "n_shift": 160,
                #                  "window": "han"},
                #                 {"type": "foo", ...}, ...]}
                conf = json.load(f)
                assert "process" in conf, conf
                # Find stft setting
                for p in conf["process"]:
                    if p["type"] == "stft":
                        istft = IStft(
                            win_length=p["win_length"],
                            n_shift=p["n_shift"],
                            window=p.get("window", "hann"),
                        )
                        logging.info(
                            "stft is found in {}. "
                            "Setting istft config from it\n{}".format(
                                preprocess_conf, istft
                            )
                        )
                        frame_shift = p["n_shift"]
                        break
        if istft is None:
            # Set from command line arguments
            istft = IStft(
                win_length=args.istft_win_length,
                n_shift=args.istft_n_shift,
                window=args.istft_window,
            )
            logging.info(
                "Setting istft config from the command line args\n{}".format(istft)
            )

    # sort data
    keys = list(js.keys())
    feat_lens = [js[key]["input"][0]["shape"][0] for key in keys]
    sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
    keys = [keys[i] for i in sorted_index]

    def grouper(n, iterable, fillvalue=None):
        kargs = [iter(iterable)] * n
        return zip_longest(*kargs, fillvalue=fillvalue)

    num_images = 0
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)

    for names in grouper(args.batchsize, keys, None):
        batch = [(name, js[name]) for name in names]

        # May be in time region: (Batch, [Time, Channel])
        org_feats = load_inputs_and_targets(batch)[0]
        if transform is not None:
            # May be in time-freq region: : (Batch, [Time, Channel, Freq])
            feats = transform(org_feats, train=False)
        else:
            feats = org_feats

        with torch.no_grad():
            enhanced, mask, ilens = model.enhance(feats)

        for idx, name in enumerate(names):
            # Assuming mask, feats : [Batch, Time, Channel. Freq]
            #          enhanced    : [Batch, Time, Freq]
            enh = enhanced[idx][: ilens[idx]]
            mas = mask[idx][: ilens[idx]]
            feat = feats[idx]

            # Plot spectrogram
            if args.image_dir is not None and num_images < args.num_images:
                import matplotlib.pyplot as plt

                num_images += 1
                ref_ch = 0

                plt.figure(figsize=(20, 10))
                plt.subplot(4, 1, 1)
                plt.title("Mask [ref={}ch]".format(ref_ch))
                plot_spectrogram(
                    plt,
                    mas[:, ref_ch].T,
                    fs=args.fs,
                    mode="linear",
                    frame_shift=frame_shift,
                    bottom=False,
                    labelbottom=False,
                )

                plt.subplot(4, 1, 2)
                plt.title("Noisy speech [ref={}ch]".format(ref_ch))
                plot_spectrogram(
                    plt,
                    feat[:, ref_ch].T,
                    fs=args.fs,
                    mode="db",
                    frame_shift=frame_shift,
                    bottom=False,
                    labelbottom=False,
                )

                plt.subplot(4, 1, 3)
                plt.title("Masked speech [ref={}ch]".format(ref_ch))
                plot_spectrogram(
                    plt,
                    (feat[:, ref_ch] * mas[:, ref_ch]).T,
                    frame_shift=frame_shift,
                    fs=args.fs,
                    mode="db",
                    bottom=False,
                    labelbottom=False,
                )

                plt.subplot(4, 1, 4)
                plt.title("Enhanced speech")
                plot_spectrogram(
                    plt, enh.T, fs=args.fs, mode="db", frame_shift=frame_shift
                )

                plt.savefig(os.path.join(args.image_dir, name + ".png"))
                plt.clf()

            # Write enhanced wave files
            if enh_writer is not None:
                if istft is not None:
                    enh = istft(enh)
                else:
                    enh = enh

                if args.keep_length:
                    if len(org_feats[idx]) < len(enh):
                        # Truncate the frames added by stft padding
                        enh = enh[: len(org_feats[idx])]
                    elif len(org_feats) > len(enh):
                        padwidth = [(0, (len(org_feats[idx]) - len(enh)))] + [
                            (0, 0)
                        ] * (enh.ndim - 1)
                        enh = np.pad(enh, padwidth, mode="constant")

                if args.enh_filetype in ("sound", "sound.hdf5"):
                    enh_writer[name] = (args.fs, enh)
                else:
                    # Hint: To dump stft_signal, mask or etc,
                    # enh_filetype='hdf5' might be convenient.
                    enh_writer[name] = enh

            if num_images >= args.num_images and enh_writer is None:
                logging.info("Breaking the process.")
                break
