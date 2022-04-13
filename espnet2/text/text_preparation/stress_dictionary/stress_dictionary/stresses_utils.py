#!/usr/bin/env python3.7
"""Some useful utilities for managing stresses dictionaries.

Usage:
    stresses_utils.py set <input-file> <output-file> [--missed <missed-output-file>]
    stresses_utils.py unite <dict-file>... --output <output-dict-file>
    stresses_utils.py patronymic <input-file> --output <output-dict-file> [--missed <missed-output-file>]
    stresses_utils.py (--help | -h)
    stresses_utils.py --version
"""

import re
import sys
import docopt
import stress_dictionary as sd
import stress_dictionary.stress_lib as lib
import stress_dictionary.stress_russian_patronymic as rp

#import stresses_lib as lib


def set_stresses(input_file, output_file, missed_output_file):
    """Load file with words and set stresses for words.

    1. input_file         - input file with words;
    2. output_file        - output file with stresses;
    3. missed_output_file - output file with not-found words.
    """
    vowels = re.compile("[" + lib.RUSSIAN_VOWELS + "]")
    stresses = lib.load_package_dicts('wiki')
    stressed_words = {}
    not_found_words = []
    with open(input_file, 'r') as f:
        for line in f:
            word = line.strip().lower()
            strs = stresses.get(word, None)
            # TODO unite this algorithm with StressesDictionary basic algorithm
            if strs:
                stressed_words[word] = strs
            else:
                yo = word.find('ё') + 1
                if yo > 0:
                    stressed_words[word] = [yo]
                else:
                    vs = [m.start() + 1 for m in vowels.finditer(word)]
                    if len(vs) == 1:
                        # only one vowel
                        stressed_words[word] = [vs[0]]
                    elif len(vs) == 2 and vs[0] + 1 == vs[1]:
                        # only two vowels, place stress on last
                        stressed_words[word] = [vs[1]]
                    else:
                        # unknown position
                        not_found_words.append(word)
                        pass
    lib.save_stresses(stressed_words, output_file)
    if missed_output_file:
        with open(missed_output_file, 'w') as f:
            f.write('\n'.join(not_found_words))


def unite_stresses(dict_filenames, output_filename):
    """ Unites some dictionaries into one.

    1. dict_filenames  - list of dictionary filenames, needed to unite.
    2. output_filename - output dictionary filename
    """
    united = lib.load_stresses_file(dict_filenames[0])
    for fn in dict_filenames[q:]:
        d = lib.load_stresses_file(fn)
        for k, p in d.items():
            np = united.get(k, set()).union(p)
            united[k] = np
    lib.save_stresses(united, output_filename)


def set_patronymic_stresses(input_filename, dicts_filenames, output_dict_filename, missed_output_filename):
    # print(f'=== "{input_filename}" "{dicts_filenames}" "{output_dict_filename}" "{missed_output_filename}"')
    # return
    """ Generate patronymics based on dictionary and list from input file.

        For each line in `input_filename` both male and female patronymics are generated.
        Then it is tried to set stress (based on original name and dictionary from `dict_filename`).
        If it was successful, they adds to output dictionary (which saved to `output_dict_filename`).
        If not, they adds to file `output_unknown_filename`.

    1. input_filename - list with male names.
    2. dicts_filenames  - dictionaries for searching stresses.
    3. output_dict_filename - filename for dictionary with generated patronymics.
    4. missed_output_filename - filename for list of patronymics with unknown stress.
    """
    stresses = sd.StressDictionary(dicts_filenames)
    not_found_stresses = []
    patronymic_dict = {}
    with open(input_filename, 'r') as inf:
        for line in inf:
            male_name = line.strip().lower()
            patronymics = rp.make_patronymics(stresses, male_name)
            if patronymics:
                stresses_positions = stresses.stress_positions(male_name)
                if stresses_positions:
                    for patronymic in patronymics:
                        patronymic_dict[patronymic] = [stresses_positions[-1] + 1]
                else:
                    not_found_stresses += patronymics
            else:
                # print(male_name)
                pass
    lib.save_stresses(patronymic_dict, output_dict_filename)
    if missed_output_filename:
        with open(missed_output_filename, 'w') as f:
            f.write('\n'.join(not_found_stresses))


def main():
    """ Main processing function.
    """
    args = docopt.docopt(__doc__, version=sd.__version__)
    if args['set']:
        set_stresses(args['<input-file>'], args['<output-file>'], args['<missed-output-file>'])
    elif args['unite']:
        unite_stresses(args['<dict-file>'])
    if args['patronymic']:
        set_patronymic_stresses(args['<input-file>'], [], args['<output-dict-file>'], args['<missed-output-file>'])
    else:
        print(args)
        assert False, "Unknown command"


if __name__ == '__main__':
    main()
