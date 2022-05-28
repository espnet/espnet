#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import argparse
import codecs
import os
from multiprocessing import Pool

import numpy as np
import scipy.io.wavfile as wf
from nara_wpe.utils import istft, stft
from nara_wpe.wpe import wpe_v8 as wpe


def wpe_worker(
    wav_scp,
    data_root="MISP_121h",
    output_root="MISP_121h_WPE_",
    processing_id=None,
    processing_num=None,
):
    sampling_rate = 16000
    iterations = 5
    stft_options = dict(
        size=512,
        shift=128,
        window_length=None,
        fading=True,
        pad=True,
        symmetric_window=False,
    )
    with codecs.open(wav_scp, "r") as handle:
        lines_content = handle.readlines()
    wav_lines = [*map(lambda x: x[:-1] if x[-1] in ["\n"] else x, lines_content)]
    for wav_idx in range(len(wav_lines)):
        if processing_id is None:
            processing_token = True
        else:
            if wav_idx % processing_num == processing_id:
                processing_token = True
            else:
                processing_token = False
        if processing_token:
            file_list = wav_lines[wav_idx].split(" ")
            name, wav_list = file_list[0], file_list[1:]
            file_exist = True
            for wav_path in wav_list:
                file_exist = file_exist and os.path.exists(
                    wav_path.replace(data_root, output_root)
                )
                if not file_exist:
                    break
            if not file_exist:
                print("wait to process {} : {}".format(wav_idx, wav_list[0]))
                signal_list = []
                for f in wav_list:
                    _, data = wf.read(f)
                    if data.dtype == np.int16:
                        data = np.float32(data) / 32768
                    signal_list.append(data)
                min_len = len(signal_list[0])
                max_len = len(signal_list[0])
                for i in range(1, len(signal_list)):
                    min_len = min(min_len, len(signal_list[i]))
                    max_len = max(max_len, len(signal_list[i]))
                if min_len != max_len:
                    for i in range(len(signal_list)):
                        signal_list[i] = signal_list[i][:min_len]
                y = np.stack(signal_list, axis=0)
                Y = stft(y, **stft_options).transpose(2, 0, 1)
                Z = wpe(Y, iterations=iterations, statistics_mode="full").transpose(
                    1, 2, 0
                )
                z = istft(Z, size=stft_options["size"], shift=stft_options["shift"])
                for d in range(len(signal_list)):
                    store_path = wav_list[d].replace(data_root, output_root)
                    if not os.path.exists(os.path.split(store_path)[0]):
                        os.makedirs(os.path.split(store_path)[0], exist_ok=True)
                    tmpwav = np.int16(z[d, :] * 32768)
                    wf.write(store_path, sampling_rate, tmpwav)
            else:
                print("file exist {} : {}".format(wav_idx, wav_list[0]))
    return None


def wpe_manager(
    wav_scp, processing_num=1, data_root="MISP_121h", output_root="MISP_121h_WPE_"
):
    if processing_num > 1:
        pool = Pool(processes=processing_num)
        for i in range(processing_num):
            pool.apply_async(
                wpe_worker,
                kwds={
                    "wav_scp": wav_scp,
                    "processing_id": i,
                    "processing_num": processing_num,
                    "data_root": data_root,
                    "output_root": output_root,
                },
            )
        pool.close()
        pool.join()
    else:
        wpe_worker(wav_scp, data_root=data_root, output_root=output_root)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_wpe")
    parser.add_argument(
        "wav_scp",
        type=str,
        default="./local/tmp/wpe.scp",
        help="list file of wav, format is scp",
    )
    parser.add_argument(
        "data_root", type=str, default="wpe", help="input misp data root"
    )
    parser.add_argument(
        "output_root", type=str, default="wpe", help="output wpe data root"
    )
    parser.add_argument("-nj", type=int, default="1", help="number of process")
    args = parser.parse_args()
    print("wavfile=", args.wav_scp)
    print("processingnum=", args.nj)
    wpe_manager(
        wav_scp=args.wav_scp,
        processing_num=args.nj,
        data_root=args.data_root,
        output_root=args.output_root,
    )
