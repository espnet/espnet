"""Modified from the official script:
https://github.com/bytedance/neurst/tree/master/datasets/GigaST
"""

import argparse
import copy
import json


def preprocess_gigaspeech_text(text: str) -> str:
    garbage_tags = [
        "<SIL>",
        "<MUSIC>",
        "<NOISE>",
        "<OTHER>",
    ]
    for g in garbage_tags:
        if g in text:
            return ""

    text = text.replace(" <COMMA>", ",")
    text = text.replace(" <PERIOD>", ".")
    text = text.replace(" <QUESTIONMARK>", "?")
    text = text.replace(" <EXCLAMATIONPOINT>", "!")

    assert "<" not in text and ">" not in text, text
    return text.lower()


def preprocess_gigast_text(text: str) -> str:
    invalid = [
        "<--plhd--8/>",
        "<i>",
        "</i>",
        "</I>",
        "</u>",
    ]
    for t in invalid:
        text = text.replace(t, "")

    text = text.strip("<>")  # remove < or > from beginning and end
    text = " ".join(text.split())

    assert "<" not in text and ">" not in text, text
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gigaspeech_file", type=str, required=True, help="The GigaSpeech.json file."
    )
    parser.add_argument(
        "--gigast_file", type=str, required=True, help="The GigaST data file."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="The output file."
    )
    args = parser.parse_args()

    aid_sid_seg = {}
    with open(args.gigast_file) as fp:
        gigast = json.load(fp)
    for audio in gigast.pop("audios"):
        for segment in audio["segments"]:
            sid = segment["sid"]
            aid = sid.split("_")[0]
            if aid not in aid_sid_seg:
                aid_sid_seg[aid] = {}
            aid_sid_seg[aid][sid] = segment

    data = copy.deepcopy(gigast)
    data["audios"] = []

    with open(args.gigaspeech_file) as fp:
        gigaspeech = json.load(fp)

    for audio in gigaspeech["audios"]:
        aid = audio["aid"]
        matched_elems = aid_sid_seg.get(aid, None)
        if matched_elems is None:
            continue
        segments = audio.pop("segments")
        this_audio = copy.deepcopy(audio)
        this_audio["segments"] = []
        for segment in segments:
            sid = segment["sid"]
            if sid not in matched_elems:
                continue

            this_segment = copy.deepcopy(segment)

            asr_text = preprocess_gigaspeech_text(this_segment["text_tn"])
            assert asr_text, asr_text

            text_tn = preprocess_gigast_text(matched_elems[sid]["text_raw"])
            if not text_tn:
                continue

            this_segment["text_asr"] = asr_text
            this_segment["text_raw"] = ""
            this_segment["text_tn"] = text_tn
            if "extra" in matched_elems[sid]:
                this_segment["extra"] = matched_elems[sid]["extra"]

            this_audio["segments"].append(this_segment)
        data["audios"].append(this_audio)
    with open(args.output_file, "w") as fw:
        json.dump(data, fw, indent=4)
