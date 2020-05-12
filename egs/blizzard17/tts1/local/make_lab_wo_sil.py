#!/usr/bin/env python3

# Copyright 2018 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import csv
import sys

import librosa
import numpy as np


def get_vp(in_lab_file):
    # initialize output
    voice_part_old = []
    utt_array = []  # uttrance array w/o '#'

    # get voice_part & utt_array
    with open(in_lab_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for x in reader:
            if x[2] != "#":
                voice_part_old.append(np.float(x[0]))
                voice_part_old.append(np.float(x[1]))
                utt_array.append(x[2])
    voice_part_old = np.reshape(voice_part_old, (-1, 2))

    return voice_part_old, utt_array


def ignore_sil(voice_part_ana, sec=0.1):
    # set length of voice_part
    avn = len(voice_part_ana)  # Number of Analysed Voice part

    # check short silent
    check_list = np.ones(
        avn + 1, np.int64
    )  # 1:time to use, 0:time to ignore (=short silence)
    for i in range(avn - 1):
        diff = voice_part_ana[i + 1][0] - voice_part_ana[i][1]
        if diff < sec:
            check_list[i + 1] = 0
    st_list = check_list[0:-1]
    ed_list = check_list[1:]

    # initialize counter
    stn = 0
    edn = 0

    # initialize output
    tvn = sum(st_list)  # Number of Tmp Voice part
    voice_part_tmp = np.zeros((tvn, 2), np.float)

    # ignore short silence
    for i in range(avn):
        if st_list[i] == 1:
            voice_part_tmp[stn][0] = voice_part_ana[i][0]
            stn = stn + 1
        if ed_list[i] == 1:
            voice_part_tmp[edn][1] = voice_part_ana[i][1]
            edn = edn + 1

    return voice_part_tmp


def separete_vp(voice_part_old, voice_part_tmp):
    # set length of voice_part
    ovn = len(voice_part_old)  # Number of Old Voice part
    tvn = len(voice_part_tmp)  # Number of Tmp Voice part

    # initialize output
    voice_part_add = []
    del_list = []
    # pick up the additional separater
    for i in range(ovn):
        for j in range(tvn):
            if (voice_part_tmp[j][0] < voice_part_old[i][0]) and (
                voice_part_old[i][0] < voice_part_tmp[j][1]
            ):
                voice_part_add.append(voice_part_old[i][0])
                voice_part_add.append(voice_part_tmp[j][1])
                if not np.any(del_list == j):
                    del_list.append(j)
                    if (voice_part_tmp[j][0] < voice_part_old[i][1]) and (
                        voice_part_old[i][1] < voice_part_tmp[j][1]
                    ):
                        voice_part_add.append(voice_part_old[i][0])
                        voice_part_add.append(voice_part_old[i][1])

            if (voice_part_tmp[j][0] < voice_part_old[i][1]) and (
                voice_part_old[i][1] < voice_part_tmp[j][1]
            ):
                voice_part_add.append(voice_part_tmp[j][0])
                voice_part_add.append(voice_part_old[i][1])
                if not np.any(del_list == j):
                    del_list.append(j)

    # delite old separater
    voice_part_res = np.delete(voice_part_tmp, del_list, 0)
    rvn = len(voice_part_res)  # Number of Resodual Voice part

    # merge the additional separater
    for i in range(rvn):
        voice_part_add.append(voice_part_res[i][0])
        voice_part_add.append(voice_part_res[i][1])

    return np.sort(np.reshape(voice_part_add, (-1, 2)), axis=0)


def create_vp(voice_part_old, voice_part_tmp):
    # set length of voice_part
    ovn = len(voice_part_old)  # Number of Old Voice part
    tvn = len(voice_part_tmp)  # Number of Tmp Voice part

    # initialize output
    voice_part_new = np.zeros((ovn, 2), np.float)

    # merge voice_part & skip non_voice_part
    for i in range(ovn):
        for j in range(tvn):
            if (voice_part_old[i][0] <= voice_part_tmp[j][0]) and (
                voice_part_tmp[j][1] <= voice_part_old[i][1]
            ):
                if voice_part_new[i][0] == 0.0:
                    voice_part_new[i][0] = voice_part_tmp[j][0]
                voice_part_new[i][1] = voice_part_tmp[j][1]

    # check error
    for i in range(ovn):
        for j in range(2):
            if voice_part_new[i][j] == 0.0:
                sys.stderr.write("Error: Element[%d][%d] is zero.\n" % (i, j))

    return voice_part_new


def write_new_lab(out_lab_file, dur, voice_part_new, utt_array):
    # set length of voice_part
    nvn = len(voice_part_new)  # Number of New Voice part

    with open(out_lab_file, mode="w") as f:
        for i in range(nvn):
            if i == 0:
                # Head
                f.write("%.6f\t%.6f\t%s\n" % (0, np.float(voice_part_new[i][0]), "#"))
            else:
                # Body of silence
                f.write(
                    "%.6f\t%.6f\t%s\n"
                    % (
                        np.float(voice_part_new[i - 1][1]),
                        np.float(voice_part_new[i][0]),
                        "#",
                    )
                )
            # Body of voice
            f.write(
                "%.6f\t%.6f\t%s\n"
                % (
                    np.float(voice_part_new[i][0]),
                    np.float(voice_part_new[i][1]),
                    utt_array[i],
                )
            )
        # Tail
        f.write("%.6f\t%.6f\t%s\n" % (np.float(voice_part_new[i][1]), dur, "#"))


def compare_vp(voice_part_old, voice_part_new):
    vn = len(voice_part_old)
    base_dur = 1.2

    for i in range(vn):
        diff_st = voice_part_new[i][0] - voice_part_old[i][0]
        diff_ed = voice_part_old[i][1] - voice_part_new[i][1]
        dur_old = voice_part_old[i][1] - voice_part_old[i][0]
        dur_new = voice_part_new[i][1] - voice_part_new[i][0]
        diff_dur = dur_old - dur_new

        if 1.0 < diff_st:
            sys.stderr.write(
                "Warning: StDiff[%4f][%d] is bigger than 1.\n" % (diff_st, i)
            )
        if 1.0 < diff_ed:
            sys.stderr.write(
                "Warning: EdDiff[%4f][%d] is bigger than 1.\n" % (diff_ed, i)
            )
        if dur_new < 0.5:
            sys.stderr.write(
                "Warning: NewDur(%4f)[%d] is smaller than 0.5.\n" % (dur_new, i)
            )
        if (diff_dur < 0) or (base_dur < diff_dur):
            sys.stderr.write(
                "Warning: Diff(%4f)[%d] is out of 0<x<%.2f.\n" % (diff_dur, i, base_dur)
            )


def main():
    args = sys.argv
    in_wav_file = args[1]
    in_lab_file = args[2]
    out_lab_file = args[3]

    # in_wav_file open
    wav_form, fs = librosa.core.load(in_wav_file)  # wave form, sampling frequency
    dur = len(wav_form) / fs  # duration

    # extract voice_part_ana from wav_file
    voice_part_ana = (
        librosa.effects.split(wav_form, top_db=40, frame_length=2048, hop_length=512)
        / fs
    )

    # in_lab_file open & get voice_part_old from lab_file
    voice_part_old, utt_array = get_vp(in_lab_file)

    # connect voice_part_ana to ignore short silent
    voice_part_tmp = ignore_sil(voice_part_ana, sec=0.01)

    # separate voice_part based on lab
    voice_part_tmp = separete_vp(voice_part_old, voice_part_tmp)

    # create voice_part_new by merging voice_parts and skipping non_voice_part
    voice_part_new = create_vp(voice_part_old, voice_part_tmp)

    # compare voice_part_old and voice_part_new (for debug)
    # compare_vp(voice_part_old, voice_part_new)

    # write duration of new_lab
    write_new_lab(out_lab_file, dur, voice_part_new, utt_array)


if __name__ == "__main__":
    main()
