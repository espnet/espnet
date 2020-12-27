import re
import pinyin_transform as pt

def main():
    # Taigi
    syllable_map = pt.SYLLABLE_MAP(source="Dai-Lor", target="IPA")
    tone_map = pt.TONE_MAP(source="Dai-Lor", target="mix")
    input = "beh_4"
    syllable, tone = input.split("_")
    tone = tone_map.transform(tone)
    ipa = syllable_map.transform(syllable).replace("-", " ").format(tone)
    print(input, ipa)

    # Mandarin
    syllable_map = pt.SYLLABLE_MAP(source="Zhuyin", target="IPA")
    input = "ㄅㄧㄢˋ"
    tone = pt.get_chewing_tone(input)
    syllable = re.sub("[ˊˇˋ˙]", "", input)
    ipa = syllable_map.transform(syllable).replace("-", " ").format(tone)
    print(input, ipa)

if __name__ == "__main__":
    main()
