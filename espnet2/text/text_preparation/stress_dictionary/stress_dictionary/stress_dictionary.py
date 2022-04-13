""" StressDictionary class wich allow to add stresses to text.
"""
import json
import re
import os
import Stemmer
# pip install "PyStemmer@git+https://github.com/Desklop/pystemmer@2.0.0"
import stress_dictionary.stress_lib as lib


WORDS_PATTERN = re.compile('[а-яёА-ЯЁ+][-а-яёА-ЯЁ+]*')


class StressDictionary:
    """ Object to set stresses in text.

        Private fields:
        1. __stemmer - Stemmer for words
        2. __stresses - Stresses dictonary:
           'word' -> 'stress position'
        3. __stem_stresses - Stresses dictonary for stemmed words:
          'stemmed word' -> 'stress position in stemmed word'
    """

    """ Dictionary of all stresses.
        It is a dict object with list of stress positions.
    """
    __stresses = {}

    def __init__(self, dict_filenames=[], dict_names_regex='.?'):
        """ Initialize object. Loads all listed in
            `dict_filenames` dictionaries for futher processing.

            1. dict_filenames   - list of JSON dictionaries to load.
            1. dict_names_regex - regex of dictionary names to load.
        """

        self.__load_package_dicts(dict_names_regex)
        self.__load_stress_dicts(dict_filenames)
        self.__stemmer = Stemmer.Stemmer('russian')
        self.__make_stems_dict()

    def stress(self, text):
        """ Add stresses to text

            1. text - text in wich need to set stresses.
               It may contain already stressed words
            2. return text with stresses using mark '+'
        """
        result = text
        accumulated_shift = 0
        for stress_position in self.stress_positions(text):
            stress_mark_position = stress_position + accumulated_shift + 1
            result = result[:stress_mark_position] + '+' + result[stress_mark_position:]
            accumulated_shift += 1
        return result

    def stress_positions(self, text):
        """ Returns stresses in text.

            1. text - text in wich need to set stresses.
               It may contain already stressed words
            2. return list of stress positions
        """
        stress_positions = []
        result = text
        accumulated_shift = 0
        for m in WORDS_PATTERN.finditer(text):
            word = m.group(0).lower()
            if '+' in word:
                # already processed word
                continue
            if len(word) == 1:
                # skip single vowel
                continue
            stresses_pos = self.__get_word_stresses(word)
            # some stresses found
            for stress_pos in stresses_pos:
                stress_positions.append(m.start() + stress_pos - 1)
        return stress_positions

    def remove_stresses(self, text):
        """ Remove stresses from text

            1. text - text to remove stresses
            2. return text without stresses
        """
        return text.replace('+', '')

    def add_stress(self, word, stress_position, dictfilename=None):
        """ Add example of stress into dictonary.

            1. word - stressed word
            2. stress_position - stress position for that word
            3. dictfilename - dictionary file name to save stress
        """
        self.__stresses[word] = [stress_position]
        stemmed_word = self.__stemmer.stemWord(word)
        if len(stemmed_word) > stress_position:
            self.__stem_stresses[stemmed_word] = [stress_position]
        def add_word(d):
            d[word] = [stress_position]
            return d
        self.__modify_dictionary_in_file(dictfilename, add_word)

    def delete_stress(self, word, dictfilename=None):
        """ Add example of stress into dictonary.

            1. word - stressed word
            2. dictfilename - dictionary file name to save stress
        """
        self.__stresses.pop(word, None)
        self.__stem_stresses.pop(self.__stemmer.stemWord(word), None)
        def del_word(d):
            d.pop(word, None)
            return d
        self.__modify_dictionary_in_file(dictfilename, del_word)

    def get_all_stresses(self):
        """ Returns stresses dictionary
        """
        return __stresses

    # private:

    def __load_package_dicts(self, dictnames_regex):
        """ Loads word stresses from packages resources.

        1. dictnames_regex - regex to match loaded dictionaries
        """
        self.__stresses = lib.load_package_dicts(dictnames_regex)

    def __load_stress_dicts(self, dict_filenames):
        """ Loads word stresses from files.

            1. dict_filenames - list of JSON dictionaries to load.
        """
        s = {}
        for fn in dict_filenames:
            with open(fn, 'r') as f:
                stresses = json.load(f)
                self.__stresses = dict(self.__stresses, **stresses)

    def __make_stems_dict(self):
        """ Make dictionary 'stem' -> 'stresses' by self.__stresses dictionary.
        """
        self.__stem_stresses = {}
        for word, stresses in self.__stresses.items():
            sw = self.__stemmer.stemWord(word)
            strs = [s for s in stresses if s <= len(sw)]
            if strs:
                self.__stem_stresses[sw] = strs

    def __get_word_stresses_basic(self, word, is_stemmed):
        """ Search stress for given word using basic ways (lookup in dictionary
            and using simple rules).

            1. word -- word to be stressed.
            2. is_stemmed -- is `word` already stemmed?
            3. return list with stress or empty list if none stresses found.
        """
        strs = self.__stresses.get(word, None)
        if strs:
            # TODO what to do with other stresses?
            return strs[:1]
        else:
            vps = lib.vowel_positions(word)
            if not is_stemmed:
                if len(vps) == 1:
                    # only one vowel in word
                    return list(vps)
                if len(vps) == 2:
                    # two consequent vowels
                    p = list(vps)
                    if p[0] + 1 == p[1]:
                        return [p[1]]
            yo_pos = [p for p, v in vps.items() if v == 'ё']
            if yo_pos:
                return yo_pos[:1]
            return []

    def __get_word_stresses(self, word):
        """ Returns stresses for given word.
            Try to reduce unknown word to known subwords if basic
            rules not worked.

            1. word - word to be stressed.
            2. return list with stress or empty list if none stresses found.
        """
        result = self.__get_word_stresses_basic(word, False)
        if result:
            return result
        else:
            # unknown word, try reducing
            subwords = word.split('-')
            if len(subwords) > 1:
                is_tail = False
                ofs = 0
                result = []
                for subword in subwords:
                    stressed_pos = self.__get_word_stresses(subword)
                    ofs += is_tail
                    if stressed_pos:
                        result.extend(map(lambda s: s + ofs, stressed_pos))
                    ofs += len(subword)
                    is_tail = True
                return result
            else:
                # try search of stemmed word
                stword = self.__stemmer.stemWord(word)
                if stword == word:
                    return []
                result = self.__get_word_stresses_basic(stword, True)
                if result:
                    return result
                # try search in stems:
                strs = self.__stem_stresses.get(stword, None)
                if strs:
                    return strs[:1]
                else:
                    return []

    def __modify_dictionary_in_file(self, dictfilename, process_stresses):
        """ Changes dictionary in file: opens it, modifies it, writes it back.

            1. dictfilename - dictionary file name
            2. process_stresses - processing function
        """
        if dictfilename:
            stresses = {}
            with open(dictfilename, 'r') as f:
                stresses = json.load(f)
            stresses = process_stresses(stresses)
            with open(dictfilename, 'w') as f:
                json.dump(stresses, f,
                    sort_keys=True, indent=4, ensure_ascii=False)

