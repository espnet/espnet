# -*- coding: UTF-8 -*-

import os
import re
import string
import sys
from argparse import ArgumentParser
from xml.dom.minidom import parse

s = "".join(chr(c) for c in range(sys.maxunicode + 1))
ws = "".join(re.findall(r"\s", s))
outtab = " " * len(ws)
trantab = str.maketrans(ws, outtab)
delset = string.punctuation
delset = delset.replace(":", "")
delset = delset.replace("'", "")


def _ToChar(text):
    new_text = ""
    for char in text:
        if char == " ":
            new_text += "<SPACE>"
        else:
            new_text += char
        new_text += " "
    return new_text


def TextRefine(text):
    text = re.sub("<i>", "", text)
    text = re.sub("</i>", "", text)
    text = re.sub(r"\.\.\.|\*|\[.*?\]", "", text.lower())
    delset_specific = delset
    remove_clear = "()=-"
    for char in remove_clear:
        delset_specific = delset_specific.replace(char, "")
    return text.translate(str.maketrans("", "", delset_specific))


def ExtractAudioID(audioname, wav_spk_info=None):
    if wav_spk_info:
        for key in wav_spk_info.keys():
            if key in audioname:
                return key
    # print("ERROR in audioname {}".format(audioname))
    return "error"


def PackZero(number, size=6):
    return "0" * (size - len(str(number))) + str(number)


def LoadWavSpeakerInfo(info_file):
    """return dict of wav: spk_list"""

    info_file = open(info_file, "r", encoding="utf-8")
    raw_info = list(map((lambda x: x.split(",")), (info_file.read()).split("\n")))
    wav_spk_info = {}
    for mapping in raw_info:
        if len(mapping) < 3:
            continue
        [wav, spk1, spk2] = mapping
        wav_spk_info[wav] = [spk1]
        if spk2 != "":
            wav_spk_info[wav] += [spk2]
    return wav_spk_info


# def LoadSpeakerDetails(speaker_details):
#     spk_details = {}
#     spk_file = open(speaker_details, "r", encoding="utf-8")
#     content = spk_file.read()
#     last_names = re.findall(r"\\last_name (.*?)\n", content, re.S)
#     first_names = re.findall(r"\\first_name (.*?)\n", content, re.S)
#     codes = re.findall(r"\\code (.*?)\n", content, re.S)
#     assert len(last_names) == len(first_names) == len(codes)
#     for last, first, code in zip(last_names, first_names, codes):
#         spk_details["%s %s" %(" ".join(first.split()), " ".join(last.split()))] = code
#     return spk_details


def LoadSpeakerDetails(speaker_details):
    spk_details = {}
    details = parse(speaker_details)
    details = details.documentElement.getElementsByTagName("last_nameGroup")
    for person in details:
        last_name = person.getElementsByTagName("last_name")[0].firstChild.data
        first_name = person.getElementsByTagName("first_name")[0].firstChild.data
        code = person.getElementsByTagName("code")[0].firstChild.data
        spk_details["%s %s" % (first_name, last_name)] = code
    return spk_details


def TimeOrderProcess(time_order_dom):
    time_order = {}
    time_slots = time_order_dom.getElementsByTagName("TIME_SLOT")
    for time_slot in time_slots:
        # convert to second based
        time_order[time_slot.getAttribute("TIME_SLOT_ID")] = (
            float(time_slot.getAttribute("TIME_VALUE")) / 1000
        )
    return time_order


def ELANProcess(afile, spk_info, spk_details):
    try:
        elan_content = parse(afile).documentElement
    except Exception:
        print("encoding failed  %s" % afile)
        return None
    time_order = TimeOrderProcess(elan_content.getElementsByTagName("TIME_ORDER")[0])
    tiers = elan_content.getElementsByTagName("TIER")
    channels = ({}, {})
    correction_channels = ({}, {})
    for tier in tiers:
        if tier.getAttribute("LINGUISTIC_TYPE_REF") not in [
            "UtteranceType",
            "Transcripción",
            "Traducción",
            "Transcripcion",
            "Traduccíón",
        ]:
            # only consider pure caption
            print("not handle tier {}".format(tier.getAttribute("LINGUISTIC_TYPE_REF")))
            continue
        try:
            spk_name = " ".join(tier.getAttribute("PARTICIPANT").strip().split())
            code = spk_details[spk_name]
        except Exception:
            print("error speaker: %s" % tier.getAttribute("PARTICIPANT").strip())
            continue
        if code not in spk_info:
            print(spk_info)
            print("error code {}".format(code))
            continue
        if tier.getAttribute("LINGUISTIC_TYPE_REF") in [
            "UtteranceType",
            "Transcripción",
            "Transcripcion",
        ]:
            channel = channels[spk_info.index(code)]
            corr_flag = False
        elif tier.getAttribute("LINGUISTIC_TYPE_REF") in ["Traducción", "Traduccíón"]:
            channel = correction_channels[spk_info.index(code)]
            corr_flag = True
        else:
            continue
        annotations = tier.getElementsByTagName("ANNOTATION")
        for anno in annotations:
            if corr_flag:
                info = anno.getElementsByTagName("REF_ANNOTATION")[0]
                ref = info.getAttribute("ANNOTATION_REF")
            else:
                info = anno.getElementsByTagName("ALIGNABLE_ANNOTATION")[0]
                seg_id = info.getAttribute("ANNOTATION_ID")
                start = time_order[info.getAttribute("TIME_SLOT_REF1")]
                end = time_order[info.getAttribute("TIME_SLOT_REF2")]
                if start == end:
                    continue
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

            if corr_flag:
                channel[ref[1:]] = text
            else:
                channel[seg_id[1:]] = [start, end, text]
    return channels, correction_channels


def TraverseData(
    sound_dir,
    annotation_dir,
    target_dir,
    mode,
    speaker_info,
    new_data_dir,
    speaker_details,
):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    segments = open(os.path.join(target_dir, "segments"), "w", encoding="utf-8")
    wavscp = open(os.path.join(target_dir, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(target_dir, "utt2spk"), "w", encoding="utf-8")
    spk2utt = open(os.path.join(target_dir, "spk2utt"), "w", encoding="utf-8")
    text = open(os.path.join(target_dir, "text.na"), "w", encoding="utf-8")
    remix_script = open(
        os.path.join(target_dir, "remix_script.sh"), "w", encoding="utf-8"
    )
    correction_text = open(os.path.join(target_dir, "text.es"), "w", encoding="utf-8")

    # get relationship
    sound_files = {}
    annotation_files = {}
    spk2utt_prep = {}

    wav_spk_info = LoadWavSpeakerInfo(speaker_info)
    spk_details = LoadSpeakerDetails(speaker_details)
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
        if afile not in sound_files.keys():
            print("no sound file found in {}".format(afile))
        spk_info = wav_spk_info[afile]
        segment_info, correction_segment_info = ELANProcess(
            afile_path, spk_info, spk_details
        )
        if segment_info is None:
            print("no segment found for {}".format(file))
            continue

        print(
            'sox -t wavpcm "%s" -c 1 -r 16000 -t wavpcm %s-L.wav remix 1'
            % (sound_files[afile], os.path.join(new_data_dir, afile)),
            file=remix_script,
        )

        print("%s-L %s-L.wav" % (afile, os.path.join(new_data_dir, afile)), file=wavscp)

        left_channel_segments, right_channel_segments = segment_info
        (
            left_channel_segments_corr,
            right_channel_segments_corr,
        ) = correction_segment_info
        for segment_number, segment in left_channel_segments.items():
            if segment_number not in left_channel_segments_corr.keys():
                continue
            # segments: start end text
            segment_id = "%s_%s-L_%s" % (spk_info[0], afile, PackZero(segment_number))
            print(
                "%s %s-L %s %s" % (segment_id, afile, segment[0], segment[1]),
                file=segments,
            )
            print("%s %s" % (segment_id, spk_info[0]), file=utt2spk)
            print("%s %s" % (segment_id, segment[2]), file=text)
            print(
                "%s %s" % (segment_id, left_channel_segments_corr[segment_number]),
                file=correction_text,
            )
            spk2utt_prep[spk_info[0]] = spk2utt_prep.get(spk_info[0], "") + " %s" % (
                segment_id
            )

        if len(right_channel_segments) > 0:
            print(
                'sox -t wavpcm "%s" -c 1 -r 16000 -t wavpcm %s-R.wav remix 2'
                % (sound_files[afile], os.path.join(new_data_dir, afile)),
                file=remix_script,
            )
            print(
                "%s-R %s-R.wav" % (afile, os.path.join(new_data_dir, afile)),
                file=wavscp,
            )
            for segment_number, segment in right_channel_segments.items():
                if segment_number not in right_channel_segments_corr.keys():
                    continue
                # segments: start end text
                segment_id = "%s_%s-R_%s" % (
                    spk_info[1],
                    afile,
                    PackZero(segment_number),
                )
                print(
                    "%s %s-R %s %s" % (segment_id, afile, segment[0], segment[1]),
                    file=segments,
                )
                print("%s %s" % (segment_id, spk_info[1]), file=utt2spk)
                print("%s %s" % (segment_id, segment[2]), file=text)
                print(
                    "%s %s" % (segment_id, right_channel_segments_corr[segment_number]),
                    file=correction_text,
                )
                spk2utt_prep[spk_info[1]] = spk2utt_prep.get(
                    spk_info[1], ""
                ) + " %s" % (segment_id)

        if correction_segment_info is None:
            print("no correction segment found for {}".format(file))
            continue

        print("successfully processing %s" % afile)
    for spk in spk2utt_prep.keys():
        print("%s %s" % (spk, spk2utt_prep[spk]), file=spk2utt)
    segments.close()
    wavscp.close()
    utt2spk.close()
    spk2utt.close()
    text.close()
    correction_text.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Process Raw data")
    parser.add_argument(
        "-w",
        dest="wav_path",
        type=str,
        help="wav path",
        default="/export/c04/jiatong/data/Puebla-Nahuatl/Sound-files-Puebla-Nahuatl",
    )
    parser.add_argument(
        "-a",
        dest="ann_path",
        type=str,
        help="annotation path",
        default="/export/c04/jiatong/data/Puebla-Nahuatl/SpeechTranslation210217",
    )
    parser.add_argument(
        "-t",
        dest="target_dir",
        type=str,
        help="target_dir",
        default="data/nahuatl_init",
    )
    parser.add_argument(
        "-i",
        dest="speaker_info",
        type=str,
        help="speaker info file dir",
        default="local/speaker_wav_mapping.csv",
    )
    parser.add_argument(
        "-m",
        dest="mode",
        type=str,
        help="transcription type",
        default="eaf",
        choices=["eaf", "trs"],
    )
    parser.add_argument(
        "-n",
        dest="new_data_dir",
        type=str,
        help="new data directory",
        default="remixed",
    )
    parser.add_argument(
        "-d",
        dest="speaker_details",
        type=str,
        help="speaker details (i.e. names to code)",
        default="local/Puebla-Nahuat-and-Totonac-consultants_for-LDC-archive.xml",
    )
    args = parser.parse_args()
    TraverseData(
        args.wav_path,
        args.ann_path,
        args.target_dir,
        mode=args.mode,
        speaker_info=args.speaker_info,
        new_data_dir=args.new_data_dir,
        speaker_details=args.speaker_details,
    )
