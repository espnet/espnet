# Stresses setter

This repo contains two modules:

* `stresses` -- set stresses in text using dictionaries;
* `extracting` -- some utilities for extracting stresses from Internet sources 
  like Wikipedia.

# `stresses` module

This module contains class `Stresses` which allow to add stresses to text
(or remove from text).

To use this module please install `PyStemmer` library
(`pip install "PyStemmer@git+https://github.com/Desklop/pystemmer@2.0.0"`).

Usage example:


```python
import stresses

# Create object and load dictonary
strs = stresses.Stresses(['wiki-stresses.json'])

# Now we can use 'add' method for add stresses
# and 'remove' for remove stresses.

txt = ('Иван Антонович Петров хочет получить от \
Евгения Пономарёва 100рублей и 12 копеек')
stressed = strs.add(txt)
print(stressed)
# Outputs:
# Ива+н Анто+нович Петро+в хо+чет получи+ть о+т Евге+ния Пономарёва+ 100рубле+й и 12 копе+ек

unstressed = strs.remove(stressed)
print(unstressed)
# Outputs:
# Иван Антонович Петров хочет получить от Евгения Пономарёва 100рублей и 12 копеек
```

Note that loading `wiki-stresses.json` dictionary take 7-8 seconds.

Module also contains tests and benchmarks.

Tests use PyTester (`pip install pytest`) and can be runned via
`py.test test/test_stresses.py`.

Benchmarks use PyPerf (`pip install pyperf`) and must be runned separately:

* `python bench/bench_test_dict_small_phrase.py` -- benchmarks small phrase on small dictionary;
* `python bench/bench_test_dict_10kb_text.py` -- benchmarks 10kb synthesized text on small dictionary;
* `python bench/bench_real_dict_10kb_text.py` -- benchmarks 10kb synthesized text on real dictionary.

Note that only `Stresses::add` function is benchmarked (dictionary loading not benchmarked).

# `extracting` module

This one contains scripts for extracting stresses from Russian Wikipedia and Wikidata and stresses dictionaries
processing utility:

* `wikidata-parsing.py` -- extractor from Wikidata.
* `wikipedia-parsing.py` -- extractor from Wikipedia.
* `stresses_utils.py` -- utility to manage stresses dictionaries. Currently it supports just two commands:
    * `stresses_utils.py set words_list.txt` -- it loads `wiki-stresses.json` dictionary and outputs to the stdout
      subdictionary which contains words from `words_list.txt`.
    * `stresses_utils.py unite dict1.json dict2.json dict3.json ...` -- it unites listed dictionaries into one
       and outputs it to the stdout. This command uses to combine dictionaries from Wikidata and Wikipedia into combined
       `wiki-stresses.json`.
* `stresses_lib.py` -- auxiliary functions.

Module also contains some dictionaries:

* `wikidata-stresses.json` -- stresses from Wikidata;
* `wikipedia-stresses.json` -- stresses from Wikipedia;
* `wiki-stresses.json` -- combined stresses from Wikidata and Wikipedia;
* `unique_names_stressed.json` -- stresses for names from file `unique_names.txt` based on `wiki-stresses.json`;
* `unique_surnames_stressed.json` -- stresses for surnames from file `unique_surnames.txt` based on `wiki-stresses.json`.

