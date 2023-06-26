#!/usr/bin/env python3
from egs2.TEMPLATE.svs1.pyscripts.utils.prep_segments import DataHandler, get_parser


class NamineDataHandler(DataHandler):
    def get_error_dict(self, input_type):
        error_dict = {}
        if input_type == "hts":
            error_dict = {
                "FACE": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (
                        labels[i].label_id == "n"
                        and labels[i - 1].label_id == "u"
                        and labels[i - 2].label_id == "r"
                    )
                    else (labels, segment, segments, False),
                ],
                "Hearts": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].label_id == "k" and labels[i - 1].label_id == "e")
                    else (labels, segment, segments, False),
                ],
                "LOVE_DISCOTHEQUE": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].label_id == "N" and labels[i - 1].label_id == "o")
                    else (labels, segment, segments, False),
                ],
                "From13th": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].label_id == "k" and labels[i - 1].label_id == "a")
                    else (labels, segment, segments, False),
                ],
                "grave": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].label_id == "n" and labels[i - 1].label_id == "u")
                    else (labels, segment, segments, False),
                ],
                "flame_heart_teto": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].label_id == "n" and labels[i - 1].label_id == "a")
                    else (labels, segment, segments, False),
                ],
                "ainouta": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (
                        labels[i].label_id == "GlottalStop"
                        and labels[i - 1].label_id == "e"
                    )
                    or (labels[i].label_id == "o" and labels[i - 1].label_id == "o")
                    else (labels, segment, segments, False),
                ],
                "GLIDE": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].label_id == "e" and labels[i - 1].label_id == "e")
                    or (labels[i].label_id == "a" and labels[i - 1].label_id == "a")
                    else (labels, segment, segments, False),
                ],
                "reverse": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].label_id == "o" and labels[i - 1].label_id == "o")
                    else (labels, segment, segments, False),
                ],
            }
        elif input_type == "xml":
            error_dict = {
                "sugar_melly": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "ゆ" and labels[i - 1].lyric == "ま")
                    else (labels, segment, segments, False),
                ],
                "FACE": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "・お" and labels[i - 1].lyric == "な")
                    or (labels[i].lyric == "の" and labels[i - 1].lyric == "る")
                    else (labels, segment, segments, False),
                ],
                "Hearts": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "い" and labels[i - 1].lyric == "る")
                    or (labels[i].lyric == "こ" and labels[i - 1].lyric == "て")
                    or (labels[i].lyric == "き" and labels[i - 1].lyric == "て")
                    else (labels, segment, segments, False),
                ],
                "entoutcas": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "う" and labels[i - 1].lyric == "も")
                    or (labels[i].lyric == "い" and labels[i - 1].lyric == "く")
                    else (labels, segment, segments, False),
                ],
                "LOVE_DISCOTHEQUE": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "か" and labels[i - 1].lyric == "わ")
                    or (labels[i].lyric == "ん" and labels[i - 1].lyric == "ろ")
                    or (labels[i].lyric == "ん" and labels[i - 1].lyric == "こ")
                    or (labels[i].lyric == "ん" and labels[i - 1].lyric == "お")
                    or (labels[i].lyric == "ん" and labels[i - 1].lyric == "しょ")
                    else (labels, segment, segments, False),
                ],
                "From13th": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "い" and labels[i - 1].lyric == "た")
                    or (labels[i].lyric == "キ" and labels[i - 1].lyric == "が")
                    or (labels[i].lyric == "く" and labels[i - 1].lyric == "だ")
                    or (labels[i].lyric == "か" and labels[i - 1].lyric == "だ")
                    or (labels[i].lyric == "き" and labels[i - 1].lyric == "な")
                    or (labels[i].lyric == "け" and labels[i - 1].lyric == "だ")
                    or (labels[i].lyric == "か" and labels[i - 1].lyric == "な")
                    or (labels[i].lyric == "キ" and labels[i - 1].lyric == "は")
                    or (labels[i].lyric == "か" and labels[i - 1].lyric == "が")
                    or (labels[i].lyric == "け" and labels[i - 1].lyric == "さ")
                    or (labels[i].lyric == "こ" and labels[i - 1].lyric == "た")
                    or (labels[i].lyric == "く" and labels[i - 1].lyric == "わ")
                    or (labels[i].lyric == "け" and labels[i - 1].lyric == "か")
                    or (labels[i].lyric == "け" and labels[i - 1].lyric == "や")
                    or (labels[i].lyric == "か" and labels[i - 1].lyric == "あ")
                    or (labels[i].lyric == "つ" and labels[i - 1].lyric == "え")
                    or (labels[i].lyric == "け" and labels[i - 1].lyric == "あ")
                    else (labels, segment, segments, False),
                ],
                "koinoowari": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "は" and labels[i - 1].lyric == "に")
                    or (labels[i].lyric == "こ" and labels[i - 1].lyric == "お")
                    or (labels[i].lyric == "・い" and labels[i - 1].lyric == "あ")
                    else (labels, segment, segments, False),
                ],
                "1st_color": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "に" and labels[i - 1].lyric == "あ")
                    else (labels, segment, segments, False),
                ],
                "machigaisagashi": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "つ" and labels[i - 1].lyric == "お")
                    or (labels[i].lyric == "こ" and labels[i - 1].lyric == "を")
                    or (labels[i].lyric == "と" and labels[i - 1].lyric == "ひ")
                    else (labels, segment, segments, False),
                ],
                "ERROR": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "な" and labels[i - 1].lyric == "も")
                    else (labels, segment, segments, False),
                ],
                "grave": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "の" and labels[i - 1].lyric == "く")
                    or (labels[i].lyric == "の" and labels[i - 1].lyric == "る")
                    else (labels, segment, segments, False),
                ],
                "finger_pencil": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "だ" and labels[i - 1].lyric == "わ")
                    or (labels[i].lyric == "ぜ" and labels[i - 1].lyric == "た")
                    or (labels[i].lyric == "た" and labels[i - 1].lyric == "よ")
                    else (labels, segment, segments, False),
                ],
                "flame_heart_teto": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "ま" and labels[i - 1].lyric == "こ")
                    or (labels[i].lyric == "・い" and labels[i - 1].lyric == "る")
                    or (labels[i].lyric == "な" and labels[i - 1].lyric == "わ")
                    or (labels[i].lyric == "り" and labels[i - 1].lyric == "た")
                    or (labels[i].lyric == "の" and labels[i - 1].lyric == "や")
                    or (labels[i].lyric == "に" and labels[i - 1].lyric == "ま")
                    or (labels[i].lyric == "の" and labels[i - 1].lyric == "た")
                    or (labels[i].lyric == "の" and labels[i - 1].lyric == "か")
                    or (labels[i].lyric == "の" and labels[i - 1].lyric == "あ")
                    or (labels[i].lyric == "ね" and labels[i - 1].lyric == "さ")
                    or (labels[i].lyric == "に" and labels[i - 1].lyric == "な")
                    else (labels, segment, segments, False),
                ],
                "ainouta": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "・あ" and labels[i - 1].lyric == "て")
                    or (labels[i].lyric == "・い" and labels[i - 1].lyric == "て")
                    or (labels[i].lyric == "お" and labels[i - 1].lyric == "お")
                    or (labels[i].lyric == "おっ" and labels[i - 1].lyric == "こ")
                    or (labels[i].lyric == "お" and labels[i - 1].lyric == "ぞ")
                    or (labels[i].lyric == "お" and labels[i - 1].lyric == "じょ")
                    or (labels[i].lyric == "お" and labels[i - 1].lyric == "ろ")
                    or (labels[i].lyric == "お" and labels[i - 1].lyric == "も")
                    or (labels[i].lyric == "お" and labels[i - 1].lyric == "の")
                    else (labels, segment, segments, False),
                ],
                "yoruniomou": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "お" and labels[i - 1].lyric == "て")
                    else (labels, segment, segments, False),
                ],
                "GLIDE": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "え" and labels[i - 1].lyric == "え")
                    or (labels[i].lyric == "あ" and labels[i - 1].lyric == "・あ")
                    or (labels[i].lyric == "あ" and labels[i - 1].lyric == "あ")
                    or (labels[i].lyric == "あ" and labels[i - 1].lyric == "ば")
                    else (labels, segment, segments, False),
                ],
                "aimaiexe": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "だ" and labels[i - 1].lyric == "お")
                    else (labels, segment, segments, False),
                ],
                "reverse": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "お" and labels[i - 1].lyric == "の")
                    or (labels[i].lyric == "お" and labels[i - 1].lyric == "そ")
                    or (labels[i].lyric == "お" and labels[i - 1].lyric == "も")
                    or (labels[i].lyric == "お" and labels[i - 1].lyric == "ほ")
                    or (labels[i].lyric == "お" and labels[i - 1].lyric == "びょ")
                    else (labels, segment, segments, False),
                ],
            }
        return error_dict


if __name__ == "__main__":
    parser, args = get_parser()
    handler = NamineDataHandler(parser, args)
    handler.process_files()
    handler.write_files()
