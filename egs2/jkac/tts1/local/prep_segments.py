#!/usr/bin/env python3

# Copyright 2021 Nagoya University (Yusuke Yasuda)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import os
import re
import sys
from collections import namedtuple

import yaml


class JKACPath(namedtuple("JKACPath", ["label_path", "wav_path", "category", "title"])):
    def recording_id(self):
        return "{}_{}".format(self.category, self.title)

    def wav_scp_str(self, sample_rate=None):
        if sample_rate is not None:
            return "{} sox {} -t wav -r {} - |".format(
                self.recording_id(), self.wav_path, sample_rate
            )
        else:
            return "{} {}".format(self.recording_id(), self.wav_path)


class JKACLabel(
    namedtuple(
        "JKACLabel",
        [
            "path",
            "chapter_id",
            "paragraph_id",
            "style_id",
            "sentence_id",
            "sentence",
            "time_begin",
            "time_end",
        ],
    )
):
    def utt_id(self):
        return "{}_{}_{}_{}_{}_{}".format(
            self.path.category,
            self.path.title,
            self.chapter_id,
            self.paragraph_id,
            self.style_id,
            self.sentence_id,
        )

    def segment_file_str(self):
        return "{} {} {:.3f} {:.3f}".format(
            self.utt_id(), self.path.recording_id(), self.time_begin, self.time_end
        )

    def kanji_sentence(self):
        return re.sub(r"\[(.+?)\|(.+?)\]", r"\1", self.sentence).replace("　", " ")

    def furigana_sentence(self):
        return re.sub(r"\[(.+?)\|(.+?)\]", r"\2", self.sentence).replace("　", " ")

    def text_file_str(self):
        return "{} {}".format(self.utt_id(), self.kanji_sentence())

    def utt2spk_str(self, speaker_id):
        return "{} {}".format(self.utt_id(), speaker_id)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare segments from text files in yaml format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=str, help="path to J-KAC corpus")
    parser.add_argument("wav_scp_path", type=str, help="path to output 'wav.scp' file")
    parser.add_argument("utt2spk_path", type=str, help="path to output 'utt2spk' file")
    parser.add_argument("text_path", type=str, help="path to output 'text' file")
    parser.add_argument(
        "segments_path", type=str, help="path to output 'segments' file"
    )
    parser.add_argument("sample_rate", type=str, help="sampling rate")
    return parser


def list_labels(root_path):
    txt_dir_path = os.path.join(root_path, "txt")
    wav_dir_path = os.path.join(root_path, "wav")
    categories = os.listdir(txt_dir_path)
    for category in categories:
        category_txt_path = os.path.join(txt_dir_path, category)
        category_wav_path = os.path.join(wav_dir_path, category)
        for label_filename in os.listdir(category_txt_path):
            if label_filename.endswith(".yaml"):
                title = label_filename.replace(".yaml", "")
                label_path = os.path.join(category_txt_path, label_filename)
                wav_path = os.path.join(category_wav_path, title + ".wav")
                yield JKACPath(
                    label_path=label_path,
                    wav_path=wav_path,
                    category=category,
                    title=title,
                )


def read_label(path):
    with open(path.label_path, "r") as f:
        label_dict = yaml.load(f, Loader=yaml.Loader)
        return parse_label(label_dict, path)


def parse_label(book_dict, path):
    for chapter_id in book_dict.keys():
        chapter = book_dict[chapter_id]
        for paragraph_id in chapter.keys():
            paragraph = chapter[paragraph_id]
            for style_id in paragraph.keys():
                style = paragraph[style_id]
                for sentence_id, sentence in enumerate(style):
                    yield JKACLabel(
                        path=path,
                        chapter_id=chapter_id,
                        paragraph_id=paragraph_id,
                        style_id=style_id,
                        sentence_id=sentence_id + 1,
                        sentence=sentence["sent"],
                        time_begin=sentence["time"][0],
                        time_end=sentence["time"][1],
                    )


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    sample_rate = None if args.sample_rate == "48000" else args.sample_rate

    with open(args.wav_scp_path, "w") as wav_scp_f, open(
        args.utt2spk_path, "w"
    ) as utt2spk_f, open(args.text_path, "w") as text_f, open(
        args.segments_path, "w"
    ) as segments_f:
        paths = list(list_labels(args.input_dir))
        paths.sort(key=lambda p: p.recording_id())
        for path in paths:
            wav_scp_f.write(path.wav_scp_str(sample_rate=sample_rate) + "\n")
            labels = list(read_label(path))
            labels.sort(key=lambda lll: lll.utt_id())
            for label in labels:
                text_f.write(label.text_file_str() + "\n")
                segments_f.write(label.segment_file_str() + "\n")
                utt2spk_f.write(label.utt2spk_str(speaker_id="JKAC") + "\n")
