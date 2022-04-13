import os
import shutil
import tempfile
import stress_dictionary as sd


TEST_DICTS = ['test/test_dict.json']
TEST_NAMES_DICTS = ['test/test_names_dict.json']


def test_add_stresses_simple():
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    assert s.stress("Мама мыла   раму") == "Ма+ма мы+ла   ра+му"


def test_stress_positions_simple():
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    #                          0123456789012345
    #                           +    +      +
    assert s.stress_positions("Мама мыла   раму") == [1, 6, 13]


def test_remove_stresses_simple():
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    assert s.remove(s.stress("Мама    мыла  раму")) == "Мама    мыла  раму"


def test_remove_stresses_simple():
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    assert s.remove_stresses(s.stress("Мама    мыла  раму")) == "Мама    мыла  раму"


def test_solid_words():
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    assert s.stress("Босс дал  Ивану Ивановичу  28долларов") == \
            "Бо+сс да+л  Ива+ну Ива+новичу  28до+лларов"
    assert s.stress("  Иван попросил 15рублей, а получил  только 12рублей, 1 рубль пропал") == \
            "  Ива+н попроси+л 15рубле+й, а получи+л  то+лько 12рубле+й, 1 ру+бль пропа+л"
    assert s.stress("...а расстояние до реки -- метров10") == \
            "...а расстоя+ние до+ реки+ -- ме+тров10"


def test_unknown_words():
    jabberwocky = """
        Варкалось. Хливкие шорьки
        Пырялись по наве,
        И хрюкотали зелюки,
        Как мюмзики в мове."""
    stressed_jabberwocky = jabberwocky.replace('Как ', 'Ка+к ') \
                                      .replace('по ', 'по+ ')
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    assert s.stress(jabberwocky) == stressed_jabberwocky


def test_hypen_simple():
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    assert s.stress("киловатт-час") == "килова+тт-ча+с"


def test_words_with_hypen():
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    assert s.stress("Кто-нибудь когда-нибудь видел светло-красный киловатт-час?") == \
        "Кто+-нибу+дь когда+-нибу+дь ви+дел светло-кра+сный килова+тт-ча+с?"


def test_stemming_simple():
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    assert s.stress("мелиорациях") == "мелиора+циях"
    assert s.stress("мелиорации") == "мелиора+ции"
    assert s.stress("расстояний") == "расстоя+ний"


def test_names_via_stemming():
    s = sd.StressDictionary(TEST_NAMES_DICTS, "don't-load-any-dicts")
    assert s.stress("иванову") == "ивано+ву"
    assert s.stress("ивановой") == "ивано+вой"


def test_words_with_yo():
    s = sd.StressDictionary([], "don't-load-any-dicts")
    assert s.stress("Ячменёв") == "Ячменё+в"
    assert s.stress("ёжик") == "ё+жик"


def test_only_one_vowel():
    s = sd.StressDictionary([], "don't-load-any-dicts")
    assert s.stress("тест") == "те+ст"
    assert s.stress("явь") == "я+вь"
    assert s.stress("сто") == "сто+"


def test_two_consequent_vowels():
    s = sd.StressDictionary([], "don't-load-any-dicts")
    assert s.stress("оон") == "оо+н"


def test_already_stressed():
    s = sd.StressDictionary(TEST_NAMES_DICTS, "don't-load-any-dicts")
    assert s.stress(s.stress("Иванов")) == "Ивано+в"
    assert s.stress(s.stress("Ива+нов паспо+рт")) == "Ива+нов паспо+рт"


def test_already_stressed_2():
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    assert s.stress(s.stress("Кто-нибудь когда-нибудь видел светло-красный киловатт-час?")) == \
        "Кто+-нибу+дь когда+-нибу+дь ви+дел светло-кра+сный килова+тт-ча+с?"


def test_not_stress_single_vowel():
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    assert s.stress("а") == "а"
    assert s.stress("и") == "и"
    assert s.stress("о") == "о"


def test_ty():
    s = sd.StressDictionary([], "don't-load-any-dicts")
    assert s.stress("ты") == "ты+"


def test_add_stresses_in_memory():
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    assert s.stress("ударение") == "ударение"
    assert s.stress("ударения") == "ударения"
    s.add_stress("ударение", 5)
    assert s.stress("ударение") == "ударе+ние"
    assert s.stress("ударения") == "ударе+ния"
    assert s.stress("ударений") == "ударе+ний"


def test_add_stresses_in_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Copy test dictionary to temp place:
        tmpstressfn = os.path.join(tmpdirname, 'tmpdict.json')
        shutil.copyfile(TEST_DICTS[0], tmpstressfn)
        # Load dictionary from temp place and add new word
        s = sd.StressDictionary([tmpstressfn], "don't-load-any-dicts")
        assert s.stress("ударение") == "ударение"
        s.add_stress("ударение", 5, tmpstressfn)
        assert s.stress("ударение") == "ударе+ние"
        # Load dictionery from temp place and ensure
        # that new word in dictionary
        s1 = sd.StressDictionary([tmpstressfn], "don't-load-any-dicts")
        assert s1.stress("ударение") == "ударе+ние"
        assert s1.stress("ударения") == "ударе+ния"
        assert s1.stress("ударений") == "ударе+ний"


def test_delete_stresses_in_memory():
    s = sd.StressDictionary(TEST_DICTS, "don't-load-any-dicts")
    assert s.stress("расстояние") == "расстоя+ние"
    assert s.stress("расстояний") == "расстоя+ний"
    s.delete_stress("расстояние")
    assert s.stress("расстояние") == "расстояние"
    assert s.stress("расстояний") == "расстояний"


def test_delete_stresses_in_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Copy test dictionary to temp place:
        tmpstressfn = os.path.join(tmpdirname, 'tmpdict.json')
        shutil.copyfile(TEST_DICTS[0], tmpstressfn)
        # Load dictionary from temp place and add new word
        s = sd.StressDictionary([tmpstressfn], "don't-load-any-dicts")
        assert s.stress("расстояние") == "расстоя+ние"
        s.delete_stress("расстояние", tmpstressfn)
        assert s.stress("расстояние") == "расстояние"
        # Load dictionery from temp place and ensure
        # that new word in dictionary
        s1 = sd.StressDictionary([tmpstressfn], "don't-load-any-dicts")
        assert s1.stress("расстояние") == "расстояние"
        assert s1.stress("расстояний") == "расстояний"


