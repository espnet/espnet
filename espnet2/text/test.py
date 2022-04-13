import re
import os
import sys

from pathlib import Path
from russian_g2p_neuro.src.model import Russian_G2P

from text_preparation.normalizers import russian_normalizer, g2p_ru_wrapper
from text_preparation.normalizers import accentizer_from_morpher_ru_wrapper


MODULE_PATH = Path(__file__).parents[0]
g2p = Russian_G2P(MODULE_PATH / 'russian_g2p_neuro/model')
letter = "квс"

def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]

def transcribe_word_list(text):
    result = []
    text = russian_normalizer(text, replacing_symbols=False, expand_difficult_abbreviations=True, use_stress_dictionary=False, use_g2p_accentor=False,
                              use_g2p=False)
    for point in ",.!?":
        text = text.replace(point, " "+point+" ")

    wordlist = text.lower().split()
    wordilst_ac = accentizer_from_morpher_ru_wrapper(text.lower()).split()
    for word, word_ac in zip(wordlist,wordilst_ac):
        prediction = g2p.predict(word)
        if word in letter:
            result.append(prediction[0]+"1")
            continue
        idx = word_ac.find("+")
        pred = []
        if idx != -1:
            idx-=1
            for ph in prediction:
                if "1" in ph:
                    ph = ph.replace("1","0")
                pred.append(ph)
            try:
                if idx>len(pred):
                    pred[idx] =pred[idx].replace("0","1")
                    pred[idx-1] =pred[idx-1].replace("0","1")
                else:
                    pred[idx] =pred[idx].replace("0","1")
            except:
                pass
        else:
            pred = prediction
        if word in ",.!?":
            result.append(word)
        else:
            ww = ' '.join(pred)
            if "1" not in ww:
                idx = ww.find("0")
                if idx != -1:
                    ww = replacer(ww, "1", idx)
            result.append(ww)
    return ' '.join(result).split()

