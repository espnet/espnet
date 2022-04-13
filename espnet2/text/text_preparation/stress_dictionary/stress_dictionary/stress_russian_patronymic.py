''' Generation of patronymic names based
'''

RUSSIAN_CONSONANTS = "бвгджзйклмнпрстфхцш"
RUSSIAN_VOWELS = "аеёиоуыэюя"


def __patronymics(name, ltr):
    """ Auxiliary function for generating suffixes for name

    1. name - part of name
    2. ltr  - 'е' for 'евич' / 'евна' or 'о' for 'ович' / 'овна' suffixes
    3. return list of generated patronymics
    """
    return [name + ltr + 'вич', name + ltr + 'вна']


def make_patronymics(stresses, name):
    """ Make patronymics from `name` using stresser in `stresses`.

    1. stresses - `StressDictionary` to find stress in original name.
    2. name     - male for make patronymics.
    3. return list `[male_patronymic, female_patronymic]` or [] if not found
    """
    last = name[-1]
    pre = name[-2]
    if len(name) <= 1:
        # Hmm, we expect that there is no such names
        return []
    elif last in RUSSIAN_CONSONANTS and last != 'й':
        if last not in "жшчщц":
            return __patronymics(name, 'о')
        else:
            return __patronymics(name, 'е')
    elif last == 'ь' and pre in RUSSIAN_CONSONANTS:
        return __patronymics(name[:-1], 'е')
    elif (pre == 'е' or pre == 'и') and last == 'я':
        return __patronymics(name[:-1], 'е')
    elif pre in RUSSIAN_VOWELS and last in RUSSIAN_VOWELS:
        return __patronymics(name, 'е')
    else:
        stress_positions = stresses.stress_positions(name)
        if not stress_positions:
            if last == 'й' and pre in RUSSIAN_VOWELS:
                # In most such cases stress is on last syllable
                stress_positions = [len(name) - 2]
            else:
                stress_positions = [-1]
        if stress_positions[-1] + 1 == len(name):
            if pre == 'ь' and last == 'я':
                return [name[:-1] + "ич", name[:-1] + "инична"]
            else:
                return __patronymics(name, 'е')
        elif last == 'й' and stress_positions[-1] + 2 == len(name):
            return __patronymics(name[:-1], 'е')
        else:
            # last non-stressed letter
            if last in RUSSIAN_VOWELS:
                if pre in "жшчщц":
                    return __patronymics(name[:-1], 'е')
                else:
                    if last in "ауы":
                        if name in ["аникита", "никита", "мина", "савва", "сила", "фока"]:
                            return [name[:-1] + "ич", name[:-1] + "ична"]
                        else:
                            return __patronymics(name[:-1], 'о')
                    elif last == 'о':
                        return __patronymics(name[:-1], 'о')
                    elif last == 'е':
                        return __patronymics(name[:-1], 'е')
                    elif last == 'и':
                        return __patronymics(name, 'е')
                    else:
                        return []
            elif pre in "еи" and last == 'й' and stress_positions[-1] + 2 < len(name):
                if len(name) == 2:
                    return __patronymics('и', 'е')
                else:
                    ppre = name[-3]
                    ppre2 = name[-4:-2]
                    is_two_consonants = len(ppre2) == 2 and \
                        ppre2[0] in RUSSIAN_CONSONANTS and \
                        ppre2[1] in RUSSIAN_CONSONANTS
                    if ppre in "кхц" or (is_two_consonants and ppre2 != "нт"):
                        return __patronymics(name[:-1], 'е')
                    else:
                        return __patronymics(name[:-2] + 'ь', 'е')
            else:
                return []
