# -*- coding: UTF-8 -*-

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


def TextRefine(text, text_format):
    text = re.sub(r"\.\.\.|\*|\[.*?\]", "", text.upper())
    delset_specific = delset
    if text_format == "underlying_full":
        remove_clear = "()=-"
        for char in remove_clear:
            delset_specific = delset_specific.replace(char, "")
    return text.translate(str.maketrans("", "", delset_specific))


def ExtractAudioID(audioname, wav_spk_info=None):
    if wav_spk_info:
        for key in wav_spk_info.keys():
            if key in audioname:
                return key
    else:
        print("ERROR in audioname")
    return "error"


def XMLRefine(input_xml, output_xml, readable=False):

    if readable:
        append = "\n"
    else:
        append = ""
    sample_processing = open(input_xml, "r", encoding="iso-8859-1")
    output = open(output_xml, "w", encoding="iso-8859-1")
    stack = ""
    stack_symbol = ""
    while True:
        line = sample_processing.readline()
        if not line:
            break
        line = line.strip()

        if stack != "":
            if len(line) > 1 and line[0] == "<":
                if line.startswith("<Who"):
                    continue
                stack += "%s</%s>%s" % (append, stack_symbol, append)
                output.write(stack)
                if line[-2:] == "/>":
                    stack = line[:-2] + ">" + append
                    stack_symbol = line[1:].split(" ")[0]
                else:
                    output.write(line + append)
                    stack, stack_symbol = "", ""
            else:
                stack += line
        elif len(line) > 2 and line[-2:] == "/>":
            stack += line[:-2] + ">" + append
            stack_symbol = line[1:].split(" ")[0]
        else:
            output.write(line + append)
    sample_processing.close()
    output.close()


def XMLProcessing(transcribe):
    DOMTree = parse(transcribe)
    trans = DOMTree.documentElement

    # get audio file information
    if trans.hasAttribute("audio_filename"):
        audio_filename = trans.getAttribute("audio_filename")
    else:
        print("audio_file error for %s" % transcribe)
        return

    # get speaker information
    speaker_info = {}
    speakers = trans.getElementsByTagName("Speakers")
    if len(speakers) < 1:
        print("no speaker found, deal individually")
    else:
        speakers = speakers[0]
        for speaker in speakers.childNodes:
            speaker_info[speaker.getAttribute("id")] = {
                "name": speaker.getAttribute("name"),
                "dialect": speaker.getAttribute("dialect"),
                "scope": speaker.getAttribute("scope"),
                "accent": speaker.getAttribute("accent"),
            }

    # process by episode/section/turn/sync
    xml_text = []
    episodes = trans.getElementsByTagName("Episode")
    for episode in episodes:
        sections = episode.getElementsByTagName("Section")
        for section in sections:
            section_list = []
            turns = section.getElementsByTagName("Turn")
            for turn in turns:
                # read as individual speech
                raw_speaker = turn.getAttribute("speaker")
                if len(raw_speaker.split(" ")) > 1:
                    # print("continue in %s" % audio_filename)
                    continue
                individual_speech = raw_speaker not in speaker_info.keys()

                syncs = turn.getElementsByTagName("Sync")
                comments = turn.getElementsByTagName("Comment")

                # remove comment nodes
                for child in comments:
                    turn.removeChild(child)
                first, last = turn.firstChild, turn.lastChild

                for sync in syncs:
                    text = sync.firstChild
                    if text is None:
                        if last.isSameNode(sync) and len(syncs) > 0:
                            section_list[-1].append(sync.getAttribute("time"))
                        continue
                    elif len(sync.childNodes) > 1:
                        # for combination of speakers
                        text = ""
                        for child in sync.childNodes:
                            text += child.data
                    else:
                        text = text.data

                    text = TextRefine(text)

                    text = text.translate(trantab)
                    start_time = sync.getAttribute("time")

                    if individual_speech:
                        spk = "None"
                    else:
                        spk = turn.getAttribute("speaker")

                    if first.isSameNode(sync):
                        if last.isSameNode(sync):
                            section_list.append(
                                [spk, text, start_time, turn.getAttribute("endTime")]
                            )
                        else:
                            section_list.append([spk, text, start_time])
                    else:
                        section_list[-1].append(start_time)
                        if last.isSameNode(sync):
                            section_list.append(
                                [spk, text, start_time, turn.getAttribute("endTime")]
                            )
                        else:
                            section_list.append([spk, text, start_time])
            xml_text.extend(section_list)
    for xml in xml_text:
        if len(xml) != 4:
            print("warning")
            print(xml)
    return audio_filename, speaker_info, xml_text


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


def LoadSpeakerDetails(speaker_details):
    spk_details = {}
    spk_file = open(speaker_details, "r", encoding="utf-8")
    content = spk_file.read()
    last_names = re.findall(r"\\last_name (.*?)\n", content, re.S)
    first_names = re.findall(r"\\first_name (.*?)\n", content, re.S)
    codes = re.findall(r"\\code (.*?)\n", content, re.S)
    assert len(last_names) == len(first_names) == len(codes)
    for last, first, code in zip(last_names, first_names, codes):
        spk_details["%s %s" % (" ".join(first.split()), " ".join(last.split()))] = code
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


def ELANProcess(afile, spk_info, spk_details, text_format):
    try:
        elan_content = parse(afile).documentElement
    except Exception:
        print("encoding failed  %s" % afile)
        return None
    time_order = TimeOrderProcess(elan_content.getElementsByTagName("TIME_ORDER")[0])
    tiers = elan_content.getElementsByTagName("TIER")
    channels = ([], [])
    for tier in tiers:
        if tier.getAttribute("LINGUISTIC_TYPE_REF") not in [
            "UtteranceType",
            "Transcription",
        ]:
            # only consider pure caption
            continue
        try:
            spk_name = " ".join(tier.getAttribute("TIER_ID").strip().split())
            if text_format == "surface":
                if "SURFACE" not in spk_name:
                    continue
                code = spk_details[spk_name[:-9]]
            else:
                if "SURFACE" in spk_name:
                    continue
                code = spk_details[spk_name]
        except Exception:
            print("error speaker: %s" % tier.getAttribute("TIER_ID").strip())
            continue
        if code not in spk_info:
            continue
        channel = channels[spk_info.index(code)]
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
            text = TextRefine(text, text_format)
            text = text.translate(trantab)
            if len(text) < 1:
                continue
            if start == end:
                continue
            channel.append([start, end, text])
    return channels


def TraverseData(
    sound_dir,
    annotation_dir,
    target_dir,
    mode,
    speaker_info,
    new_data_dir,
    speaker_details,
    text_format,
):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    segments = open(os.path.join(target_dir, "segments"), "w", encoding="utf-8")
    wavscp = open(os.path.join(target_dir, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(target_dir, "utt2spk"), "w", encoding="utf-8")
    spk2utt = open(os.path.join(target_dir, "spk2utt"), "w", encoding="utf-8")
    text = open(os.path.join(target_dir, "text"), "w", encoding="utf-8")
    name2spk = open(os.path.join(target_dir, "name2spk"), "w", encoding="utf-8")
    remix_script = open(
        os.path.join(target_dir, "remix_script.sh"), "w", encoding="utf-8"
    )

    # get relationship
    sound_files = {}
    annotation_files = {}
    spk_id = 1
    spk2utt_prep = {}
    name2spk_prep = {}

    if mode == "trs":
        if not os.path.exists(os.path.join(target_dir, "temp")):
            os.mkdir(os.path.join(target_dir, "temp"))
        audio_set = set()
        for root, dirs, files in os.walk(sound_dir):
            for file in files:
                if file[-4:] == ".wav":
                    sound_files[ExtractAudioID(file)] = os.path.join(root, file)
        for root, dirs, files in os.walk(annotation_dir):
            for file in files:
                if file[-4:] == ".trs":
                    XMLRefine(
                        os.path.join(root, file), os.path.join(target_dir, "temp", file)
                    )
                    annotation_files[file] = os.path.join(target_dir, "temp", file)
        for afile in annotation_files.keys():
            if afile == "error":
                continue
            try:
                audio_name, speakers, segment_info = XMLProcessing(
                    annotation_files[afile]
                )
            except Exception:
                print("error process %s" % annotation_files[afile])
            audio_name = audio_name.replace(" ", "")
            audio_name = ExtractAudioID(audio_name)
            if audio_name in audio_set:
                continue
            audio_set.add(audio_name)
            if "%s.wav" % audio_name not in sound_files.keys():
                print("no audio found for annotation: %s" % afile)
                continue
                # write wav.scp & segments & text files
            print(
                "%s sox -t wavpcm %s -c 1 -r 16000 -t wavpcm - |"
                % (audio_name, sound_files["%s.wav" % audio_name]),
                file=wavscp,
            )
            segment_number = 1
            temp_speaker_id = {}
            for speaker in speakers.keys():
                name2spk_prep[speakers[speaker]["name"]] = name2spk_prep.get(
                    speakers[speaker]["name"], spk_id
                )
                temp_speaker_id[speaker] = name2spk_prep[speakers[speaker]["name"]]
                if name2spk_prep[speakers[speaker]["name"]] == spk_id:
                    print(
                        "%s %s" % (speakers[speaker]["name"], PackZero(spk_id)),
                        file=name2spk,
                    )
                    spk_id += 1
            for segment in segment_info:
                # segment: [spk, text, start_time, end_time]
                if segment[0] == "None":
                    spk = spk_id
                    spk_id += 1
                else:
                    spk = temp_speaker_id[segment[0]]
                segment_id = "%s_%s_%s" % (
                    PackZero(spk),
                    audio_name,
                    PackZero(segment_number),
                )

                # skip data error
                skip = False
                for seg in segment:
                    if len(seg) < 1:
                        print("warning segment %s in %s" % (segment_id, audio_name))
                        skip = True
                if skip:
                    continue

                print(
                    "%s %s %s %s" % (segment_id, audio_name, segment[2], segment[3]),
                    file=segments,
                )
                print("%s %s" % (segment_id, PackZero(spk)), file=utt2spk)
                print("%s %s" % (segment_id, segment[1]), file=text)

                spk2utt_prep[spk] = spk2utt_prep.get(spk, "") + " %s" % (segment_id)
                segment_number += 1
            for spk in spk2utt_prep.keys():
                print("%s %s" % (spk, spk2utt_prep[spk]), file=spk2utt)
            print("successfully processing %s" % afile)
        shutil.rmtree(os.path.join(target_dir, "temp"))
    else:
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
            spk_info = wav_spk_info[afile]
            segment_info = ELANProcess(afile_path, spk_info, spk_details, text_format)
            if segment_info is None:
                continue
            left_channel_segments, right_channel_segments = segment_info

            f = sf.SoundFile(sound_files[afile])
            max_length = len(f) / f.samplerate
            print(
                'sox -t wavpcm "%s" -c 1 -r 16000 -t wavpcm %s-L.wav remix 1'
                % (sound_files[afile], os.path.join(new_data_dir, afile)),
                file=remix_script,
            )

            print(
                "%s-L %s-L.wav" % (afile, os.path.join(new_data_dir, afile)),
                file=wavscp,
            )
            segment_number = 0
            for segment in left_channel_segments:
                # segments: start end text
                segment_id = "%s_%s-L_%s" % (
                    spk_info[0],
                    afile,
                    PackZero(segment_number),
                )
                if float(segment[1]) > max_length:
                    continue
                print(
                    "%s %s-L %s %s" % (segment_id, afile, segment[0], segment[1]),
                    file=segments,
                )
                print("%s %s" % (segment_id, spk_info[0]), file=utt2spk)
                print("%s %s" % (segment_id, segment[2]), file=text)
                spk2utt_prep[spk_info[0]] = spk2utt_prep.get(
                    spk_info[0], ""
                ) + " %s" % (segment_id)
                segment_number += 1

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
                for segment in right_channel_segments:
                    # segments: start end text
                    segment_id = "%s_%s-R_%s" % (
                        spk_info[1],
                        afile,
                        PackZero(segment_number),
                    )
                    if float(segment[1]) > max_length:
                        continue
                    print(
                        "%s %s-R %s %s" % (segment_id, afile, segment[0], segment[1]),
                        file=segments,
                    )
                    print("%s %s" % (segment_id, spk_info[1]), file=utt2spk)
                    print("%s %s" % (segment_id, segment[2]), file=text)
                    spk2utt_prep[spk_info[1]] = spk2utt_prep.get(
                        spk_info[1], ""
                    ) + " %s" % (segment_id)
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
        "-w",
        dest="wav_path",
        type=str,
        help="wav path",
        default="",
    )
    parser.add_argument(
        "-a",
        dest="ann_path",
        type=str,
        help="annotation path",
        default="",
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
        default="local/Mixtec-consultant-database-unicode_2019-12-25.txt",
    )
    parser.add_argument(
        "-f",
        dest="text_format",
        type=str,
        help="text format",
        default="",
        choices=["surface", "underlying_full", "underlying_reduced", ""],
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
        text_format=args.text_format,
    )
