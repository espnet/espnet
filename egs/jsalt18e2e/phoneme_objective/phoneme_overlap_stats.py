""" Report statistics on the overlap between the phoneme sets of different languages."""

from collections import defaultdict
from pathlib import Path

text_phn_path = Path("data/tr_babel10/text.phn")

def create_langs():
    langs = set()
    for path in Path("data/").glob("et_babel*/text.phn"):
        with path.open() as f:
            for line in f:
                utter_id, *phones = line.split()
                lang = utter_id.split("-")[-1]
                langs.add(lang)
    return list(langs)

def create_lang2lines():
    lang2lines = defaultdict(list)
    for path in Path("data/").glob("et_babel*/text.phn"):
        with path.open() as f:
            for line in f:
                utter_id, *phones = line.split()
                lang = utter_id.split("-")[-1]
                lang2lines[lang].append(phones)
    return lang2lines

def create_langphones():
    langphones = defaultdict(set)
    for path in Path("data/").glob("et_babel*/text.phn"):
        with path.open() as f:
            for line in f:
                utter_id, *phones = line.split()
                lang = utter_id.split("-")[-1]
                langphones[lang] |= set(phones)
    return langphones

def jaccard(a, b):
    return len(a & b) / (len(a) + len(b) - len(a & b))

def fmt_jaccard_grid():
    langphones = create_langphones()
    langs = create_langs()

    print(("    " + "{} "*len(langs)).format(*[lang[:3] for lang in langs]))
    fmt = "{} " + "{:3.0f} "*len(langs)
    for lang_a in langs:
        jaccards = []
        for lang_b in langs:
            jaccards.append(jaccard(
                langphones[lang_a], langphones[lang_b]) * 100)
        print(fmt.format(lang_a[:3], *jaccards))
