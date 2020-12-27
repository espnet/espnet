import os
import re
from xlrd import open_workbook

def get_chewing_tone(syllable):
    if re.match(".+ˊ", syllable): tone = 2
    elif re.match(".+ˇ", syllable): tone = 3
    elif re.match(".+ˋ", syllable): tone = 4
    elif re.match(".+˙", syllable): tone = 5
    else: tone = 1
    return tone

class TONE_MAP():
    def __init__(self, source, target):
        self.tailo_tone = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
        self.ty_t_tone = self.forpa_tone = ["1", "4", "3", "7", "5", "2", "6", "9", "8"]
        self.mix_tone = ["1", "4", "3", "6", "2", "7", "8", "9", "10"]

        self.source = source
        self.target = target
        self.map = self.build_mapping()

    def build_mapping(self):
        source_tone_list, target_tone_list = [], []
        if self.source == "Dai-Lor":
            source_tone_list = self.tailo_tone
        elif self.source == "TY-T":
            source_tone_list = self.ty_t_tone
        elif self.source == "ForPA":
            source_tone_list = self.forpa_tone
        elif self.source == "mix":
            source_tone_list = self.mix_tone

        if self.target == "Dai-Lor":
            target_tone_list = self.tailo_tone
        elif self.target == "TY-T":
            target_tone_list = self.ty_t_tone
        elif self.target == "ForPA":
            target_tone_list = self.forpa_tone
        elif self.target == "mix":
            target_tone_list = self.mix_tone

        return dict(zip(source_tone_list, target_tone_list))

    def transform(self, tone):
        return self.map[tone]

class SYLLABLE_MAP():
    def __init__(self, source, target):
        self.forpa_excel = "local/pinyin2ipa/forpa_200512.xlsx"
        self.mandarin_pinyinTable = "local/pinyin2ipa/pinyin.xlsx"
        self.taigi_pinyinTable = "local/pinyin2ipa/taigi.xlsx"

        self.source = source
        self.target = target
        self.map = self.build_mapping()

    def build_mapping(self):
        pinyin_map = dict()

        if self.target == "IPA":
            if self.source  == "Zhuyin":
                pinyinTable = self.mandarin_pinyinTable
            elif self.source == "Dai-Lor":
                pinyinTable = self.taigi_pinyinTable

            with open_workbook(pinyinTable) as wb:
                sheet = wb.sheets()[0]

                for row_index in range(1, sheet.nrows):
                    row = sheet.row_values(row_index)
                    source, ipa_diphthong_tone = row[1], row[4]
                    pinyin_map[source] = ipa_diphthong_tone
        else:
            pinyin = ["TWBet", "ForPA", "Zhuyin", "Hanyu", "Dai-Lor", "TLPA", "POJ", "Uanliu", "Daiim", "TY-T", "TY-H", "TY-K"]
            col = [1, 2, 4, 6, 8, 9, 10, 11, 13, 14, 15, 16]
            pinyin2col = dict(zip(pinyin, col))

            with open_workbook(self.forpa_excel) as wb:
                sheet = wb.sheets()[0]
                nrows = sheet.nrows

                for i in range(3, nrows):
                    row = sheet.row_values(i)
                    row = [r.strip() if type(r) == str else r for r in row]
                    source_syllable, target_syllable = row[pinyin2col[self.source]], row[pinyin2col[self.target]]
                    pinyin_map[source_syllable] = target_syllable
        return pinyin_map

    def transform(self, syllable):
        if syllable in self.map: return self.map[syllable].strip()
        else:
            print("=>", syllable)
            raise Exception("No corresponding syllable in table")

"""
class TAIGI():
    # Special case
    # special_tailo = ["noh8", "berh4", "tshiuh4", "hiuh4", "gooh4", "gir2", "khngh4", "gooh4", "noh4", "thiunn2", "suainnh8", "jinn2", "tser7", "tshannh4", "tshiuh4"]
    # special_forpa = ["ner1", "bheh7", "ciu3", "hiu3", "gho3", "ghu4", "kng3", "gho3", "ner1", "tiu4", "suainnh6", "rinn2", "ze2", "cann3", "ciu3"]

    def __init__(self, type_="tailo", with_tone=True):
        self.type_ = type_
        self.with_tone = with_tone
        self.pinyinTable = "../transform_pinyin/taigi/台語pinyinTable.xlsx"
        assert os.path.exists(self.pinyinTable)

        index = {"forpa": 0, "tailo": 1}

        self.transform_table = dict()
        with open_workbook(self.pinyinTable) as wb:
            sheet = wb.sheets()[0]
            nrows = sheet.nrows

            for row_index in range(1, nrows):
                row = sheet.row_values(row_index)
                source, ipa_diphthong_tone = row[index[type_]], row[4]
                self.transform_table[source] = ipa_diphthong_tone

    def to_ipa(self, source):
        target_syllable_list = []
        syllable_list = [s for s in re.split(" |-", source) if s]
        for syllable in syllable_list:
            if self.with_tone:
                # if "_" in syllable:
                #     syllable, tone = syllable.split("_")
                # else:
                #     tone = re.search("[0-9]+", syllable)
                #     syllable = syllable.replace(tone, "")
                syllable, tone = syllable.split("_")
                tone = tone_map.transform(tone)
            else:
                syllable, tone = syllable, ""
            # try:
            target_syllable_list += [self.transform_table[syllable].format(tone)]
            # except:
                # print(syllable+tone)
                # target_syllable_list += ["Error"]
        return " ".join(target_syllable_list)


class CHINESE():
    def __init__(self, type_="tailo", with_tone=True):
        self.type_ = type_
        self.with_tone = with_tone
        self.pinyinTable = "../transform_pinyin/mandarin/華語pinyinTable-20200419.xlsx"
        assert os.path.exists(self.pinyinTable)

        index = {"forpa": 0, "chewing": 1}

        self.transform_table = dict()
        with open_workbook(self.pinyinTable) as wb:
            sheet = wb.sheets()[0]
            nrows = sheet.nrows

            for row_index in range(1, nrows):
                row = sheet.row_values(row_index)
                source, ipa_diphthong_tone = row[index[type_]], row[4]
                self.transform_table[source] = ipa_diphthong_tone

    def to_ipa(self, source):
        target_syllable_list = []
        for syllable in re.split(" ", source):
            if self.type_ == "chewing":
                if self.with_tone:
                    if re.match(".+ˊ", syllable): tone = 2
                    elif re.match(".+ˇ", syllable): tone = 3
                    elif re.match(".+ˋ", syllable): tone = 4
                    elif re.match(".+˙", syllable): tone = 5
                    else: tone = 1
                else: tone = ""
            else:
                # TODO
                tone = 0
            syllable = re.sub("[ˊˇˋ˙]", "", syllable)
            target_syllable_list += [self.transform_table[syllable].format(tone)]
        return " ".join(target_syllable_list)
"""
