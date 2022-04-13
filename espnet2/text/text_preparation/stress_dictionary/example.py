#!/usr/bin/env python
import stress_dictionary as sd

# Create object and load dictonary
strs = sd.StressDictionary(['dicts/wiki-stresses.json'])

# Now we can use 'add' method for add stresses
# and 'remove' for remove stresses.

s = 'Клим Александрович Жуков'
ss = strs.add(s)
print(f'1. Original = "{s}", stressed = "{ss}"')

s = 'Проверка ударений в тексте'
ss = strs.add(s)
print(f'2. Original = "{s}", stressed = "{ss}"')

s = ('Иван Антонович Петров хочет получить от \
Евгения Пономарёва 100рублей и 12 копеек')
ss = strs.add(s)
print(f'3. Original = "{s}", stressed = "{ss}"')

ss = strs.add(s)
sss = strs.remove(ss)
print(f'4. Original = "{s}", stressed = "{ss}", unstressed = "{sss}"')
