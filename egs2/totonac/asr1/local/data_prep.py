import os
import re
import shutil
import string
import sys
from argparse import ArgumentParser
from xml.dom.minidom import parse

import soundfile as sf

s = "".join(chr(c) for c in range(sys.maxunicode + 1))
ws = "".join(re.findall(r"\s", s))
outtab = " " * len(ws)
trantab = str.maketrans(ws, outtab)
delset = string.punctuation
delset = delset.replace(":", "")
delset = delset.replace("'", "")


def TextRefine(text):
    text = re.sub(r"\.\.\.|\*|\[.*?\]", "", text.upper())
    delset_specific = delset
    return text.translate(str.maketrans("", "", delset_specific))


def ExtractAudioID(audioname, wav_spk_info=None):
    if wav_spk_info:
        for key in wav_spk_info.keys():
            if key in audioname:
                return key
    else:
        print("ERROR in audioname")
    return "error"


def PackZero(number, size=6):
    return "0" * (size - len(str(number))) + str(number)


def LoadWavSpeakerInfo(info_file):
    """return dict of wav: spk_list"""

    info_file = open(info_file, "r", encoding="utf-8")
    raw_info = list(map((lambda x: x.split("\t")), (info_file.read()).split("\n")))
    wav_spk_info = {}
    for mapping in raw_info[1:]:
        if len(mapping) < 2:
            continue
        [spk, wav] = mapping
        wav_spk_info[wav] = spk
    return wav_spk_info


def TimeOrderProcess(time_order_dom):
    time_order = {}
    time_slots = time_order_dom.getElementsByTagName("TIME_SLOT")
    for time_slot in time_slots:
        # convert to second based
        time_order[time_slot.getAttribute("TIME_SLOT_ID")] = (
            float(time_slot.getAttribute("TIME_VALUE")) / 1000
        )
    return time_order


def ELANProcess(afile, spk_info):
    try:
        elan_content = parse(afile).documentElement
    except Exception:
        print("encoding failed  %s" % afile)
        return None
    time_order = TimeOrderProcess(elan_content.getElementsByTagName("TIME_ORDER")[0])
    tiers = elan_content.getElementsByTagName("TIER")
    channel = []
    for tier in tiers:
        if tier.getAttribute("LINGUISTIC_TYPE_REF") not in [
            "UtteranceType",
            "Transcription",
            "Transcripción",
        ]:
            # only consider pure caption
            continue

        annotations = tier.getElementsByTagName("ANNOTATION")
        for anno in annotations:
            info = anno.getElementsByTagName("ALIGNABLE_ANNOTATION")[0]
            start = time_order[info.getAttribute("TIME_SLOT_REF1")]
            end = time_order[info.getAttribute("TIME_SLOT_REF2")]
            text = ""
            childs = info.getElementsByTagName("ANNOTATION_VALUE")[0].childNodes
            for child in childs:
                if child.firstChild is not None:
                    continue
                    text += child.firstChild.data
                else:
                    text += child.data
            text = TextRefine(text)
            text = text.translate(trantab)
            if len(text) < 1:
                continue
            if start == end:
                continue
            channel.append([start, end, text])
    return channel


def TraverseData(
    sound_dir, annotation_dir, target_dir, speaker_info,
):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    segments = open(os.path.join(target_dir, "segments"), "w", encoding="utf-8")
    wavscp = open(os.path.join(target_dir, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(target_dir, "utt2spk"), "w", encoding="utf-8")
    spk2utt = open(os.path.join(target_dir, "spk2utt"), "w", encoding="utf-8")
    text = open(os.path.join(target_dir, "text"), "w", encoding="utf-8")
    name2spk = open(os.path.join(target_dir, "name2spk"), "w", encoding="utf-8")

    # get relationship
    sound_files = {}
    annotation_files = {}
    spk_id = 1
    spk2utt_prep = {}
    name2spk_prep = {}

    wav_spk_info = LoadWavSpeakerInfo(speaker_info)
    for root, dirs, files in os.walk(sound_dir):
        for file in files:
            if file[-4:] == ".wav":
                sound_files[ExtractAudioID(file, wav_spk_info)] = os.path.join(
                    root, file
                )
    for root, dirs, files in os.walk(annotation_dir):
        for file in files:
            if file[-4:] == ".eaf":
                annotation_files[ExtractAudioID(file, wav_spk_info)] = os.path.join(
                    root, file
                )
    for afile in annotation_files.keys():
        afile_path = annotation_files[afile]
        if afile == "error":
            continue
        spk_info = wav_spk_info[afile]
        if len(spk_info) < 10:
            spk_info += "-______"
        channel_segments = ELANProcess(afile_path, spk_info)
        if channel_segments is None:
            continue

        f = sf.SoundFile(sound_files[afile])
        max_length = len(f) / f.samplerate

        print(
            '%s sox -t wavpcm "%s" -c 1 -r 16000 -t wavpcm - |'
            % (afile, sound_files[afile]),
            file=wavscp,
        )
        segment_number = 0
        for segment in channel_segments:
            # segments: start end text
            segment_id = "%s_%s_%s" % (spk_info, afile, PackZero(segment_number),)
            if float(segment[1]) > max_length:
                continue
            print(
                "%s %s %s %s" % (segment_id, afile, segment[0], segment[1]),
                file=segments,
            )
            print("%s %s" % (segment_id, spk_info), file=utt2spk)
            print("%s %s" % (segment_id, segment[2]), file=text)
            spk2utt_prep[spk_info] = spk2utt_prep.get(spk_info, "") + " %s" % (
                segment_id
            )
            segment_number += 1

        print("successfully processing %s" % afile)
    for spk in spk2utt_prep.keys():
        print("%s %s" % (spk, spk2utt_prep[spk]), file=spk2utt)
    segments.close()
    wavscp.close()
    utt2spk.close()
    spk2utt.close()
    text.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Process Raw data")
    parser.add_argument(
        "-w", dest="wav_path", type=str, help="wav path", default="",
    )
    parser.add_argument(
        "-a", dest="ann_path", type=str, help="annotation path", default="",
    )
    parser.add_argument(
        "-t", dest="target_dir", type=str, help="target_dir", default="data/mixtec"
    )
    parser.add_argument(
        "-i",
        dest="speaker_info",
        type=str,
        help="speaker info file dir",
        default="local/speaker_wav_mapping_mixtec.csv",
    )
    args = parser.parse_args()
    TraverseData(
        args.wav_path, args.ann_path, args.target_dir, speaker_info=args.speaker_info,
    )
