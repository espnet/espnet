import argparse
import math
import os
import re


def float2str(number, size=6):
    number = str(math.ceil(number * 100))
    return (size - len(number)) * "0" + number


def majorityvote(line):
    count_pos = line.count("Positive")
    count_neu = line.count("Neutral")
    count_neg = line.count("Negative")
    dic = {"Positive": count_pos, "Neutral": count_neu, "Negative": count_neg}
    max_value = max(dic.values())
    # make sure max_value is unique
    keys = [key for key, value in dic.items() if value == max_value]
    label = keys[0] if len(keys) == 1 else -1
    return label


def normalize_transcript(transcript):
    # remove punctuation except apostrophes
    transcript = re.sub(r"(\.|\,|\?|\!|\-|\:|\;)", " \\1 ", transcript)
    transcript = re.sub(r"\.|\,|\?|\!|\-|\:|\;", "", transcript)
    # remove tag (e.g. [LAUGHTER])
    transcript = re.sub(r"\[.+\]", "", transcript)
    # Detect valid apostrophe cases and split those into two words
    transcript = re.sub("([a-z])'([a-z])", "\\1 '\\2", transcript)
    # Clean up special cases of standalone apostrophes
    transcript = re.sub("([a-z])' ", "\\1 ", transcript)
    # remove extra spaces
    transcript = re.sub(" +", " ", transcript)
    # remove space at the beginning of the utterance
    transcript = re.sub("^ ", "", transcript)
    return transcript


def process_data(
    target_dir, sentiment_file, text_file, wavscp_file, start_linenum, end_linenum
):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    utt2spk_list = []
    segments_list = []
    text_list = []
    wavscp_list = []
    reco2file_list = []

    with open(sentiment_file, "r", encoding="utf-8") as sf, open(
        text_file, "r", encoding="utf-8"
    ) as tf, open(wavscp_file, "r", encoding="utf-8") as wf:
        prev_spk_id_tf = 0
        prev_linenum_tf = 0
        prev_linenum_wf = 0
        for linenum, line_sf in enumerate(sf):
            if linenum >= start_linenum and linenum < end_linenum:
                # "sw02005_0[tab]0.0[tab]11.287375[tab]
                # Neutral-{Questioning}#Neutral-{No emotion}#Neutral-{No emotion}"
                utt_id_sf, start, end, sentiment = line_sf.strip().split("\t")
                # "sw02005_0" -> "sw02005"
                reco_id_sf = utt_id_sf.split("_")[0]
                label = majorityvote(sentiment)
                if label != -1:
                    tf.seek(0)
                    for linenum_tf, line_tf in enumerate(tf):
                        if linenum_tf >= prev_linenum_tf:
                            # "sw02001-A_018732-018950 oh i see uh-huh"
                            # -> "sw02001-A_018732-018950" "oh i see uh-huh"
                            utt_id_tf, transcript = line_tf.strip("\n").split(" ", 1)
                            # "sw02001-A_018732-018950" -> "sw02001-A" "018732-018950"
                            spk_id_tf, time_id = utt_id_tf.split("_")
                            # "sw02001-A" -> "sw02001"
                            reco_id_tf = spk_id_tf.split("-")[0]
                            # "018732-018950" -> "018732" "018950"
                            start_time_id, end_time_id = time_id.split("-")
                            # in case start and end time slightly differ
                            # in text and sentiment annotation
                            eps = 0.05
                            if (
                                reco_id_tf == reco_id_sf
                                and start_time_id >= float2str(float(start) - eps)
                                and start_time_id <= float2str(float(start) + eps)
                                and end_time_id >= float2str(float(end) - eps)
                                and end_time_id <= float2str(float(end) + eps)
                            ):
                                # normalize transcript
                                transcript = normalize_transcript(transcript)
                                utt2spk_list.append(
                                    "{} {}".format(utt_id_tf, spk_id_tf)
                                )
                                segments_list.append(
                                    "{} {} {:.2f} {:.2f}".format(
                                        utt_id_tf, spk_id_tf, float(start), float(end)
                                    )
                                )
                                text_list.append(
                                    "{} {} {}".format(utt_id_tf, label, transcript)
                                )

                                if prev_spk_id_tf != spk_id_tf:
                                    wf.seek(0)
                                    for linenum_wf, line_wf in enumerate(wf):
                                        if linenum_wf >= prev_linenum_wf:
                                            spk_id_wf = line_wf.split(" ")[0]
                                            if spk_id_wf == spk_id_tf:
                                                wavscp_list.append(
                                                    "{}".format(line_wf.strip("\n"))
                                                )
                                                (
                                                    reco_id_wf,
                                                    channel_id,
                                                ) = spk_id_wf.split("-")
                                                reco2file_list.append(
                                                    "{} {} {}".format(
                                                        spk_id_wf,
                                                        reco_id_wf,
                                                        channel_id,
                                                    )
                                                )
                                                prev_linenum_wf = linenum_wf
                                                break
                                prev_spk_id_tf = spk_id_tf
                                prev_linenum_tf = linenum_tf
                                break
    with open(
        os.path.join(target_dir, "utt2spk"), "w", encoding="utf-8"
    ) as utt2spk, open(
        os.path.join(target_dir, "segments"), "w", encoding="utf-8"
    ) as segments, open(
        os.path.join(target_dir, "text"), "w", encoding="utf-8"
    ) as text, open(
        os.path.join(target_dir, "wav.scp"), "w", encoding="utf-8"
    ) as wavscp, open(
        os.path.join(target_dir, "reco2file_and_channel"), "w", encoding="utf-8"
    ) as reco2file:
        utt2spk.write("\n".join(utt2spk_list) + "\n")
        segments.write("\n".join(segments_list) + "\n")
        text.write("\n".join(text_list) + "\n")
        wavscp.write("\n".join(wavscp_list) + "\n")
        reco2file.write("\n".join(reco2file_list) + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, default="data/train")
parser.add_argument("--dev_dir", type=str, default="data/dev")
parser.add_argument("--test_dir", type=str, default="data/test")
parser.add_argument("--sentiment_file", type=str, required=True)
parser.add_argument("--text_file", type=str, default="data/train/tmp/text")
parser.add_argument("--wavscp_file", type=str, default="data/train/tmp/wav.scp")

args = parser.parse_args()

# Split into train, dev, test
# Note that there is no "official" split provided.
# Using the proportion of train 90%, dev 5%, test 5% as in
# https://arxiv.org/pdf/1911.09762.pdf
print("start train file preparation...this may take a while")
process_data(
    args.train_dir, args.sentiment_file, args.text_file, args.wavscp_file, 0, 47056
)
print("start dev file preparation")
process_data(
    args.dev_dir, args.sentiment_file, args.text_file, args.wavscp_file, 47056, 49673
)
print("start test file preparation")
process_data(
    args.test_dir, args.sentiment_file, args.text_file, args.wavscp_file, 49673, 52293
)

print("Successfully finished text, utt2spk, segments, wavescp preparation")
