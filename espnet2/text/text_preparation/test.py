import sys
sys.path.append("/Users/valentinzykov/espnet2/espnet2/text/")
sys.path.append("/Users/valentinzykov/espnet2/espnet2/text/text_preparation/stress_dictionary")
sys.path.append("/Users/valentinzykov/espnet2/espnet2/text/text_preparation/yo_restorer")


import accentizer
from stressrnn import StressRNN
from typing import List, Dict, Any
from text_normalization import normalize_text, normalize_text_len, LANGUAGES, ctc_symbol_to_id
from text_and_sequences import text_to_sequence, sequence_to_text, get_symbols_length, get_ctc_symbols_length


def accentizer_from_morpher_wrapper(text: str, stress_placement_obj: Any) -> str:
    ''' Обёртка для модуля расстановки ударений accentizer от morpher.ru для унификации его использования в тестах. '''

    stress_symbol = '+'

    # Токенизатор (как отдельный, так и встроенный в метод annotate) не понимает символ ударения '+' и разбивает слова на части по этому символу
    # Для исправления этого было добавлено объединение слов по символу ударения после токенизатора с последующим их исключением из обработки
    tokens = list(accentizer.Tokenizer.tokenize(text))

    i = 0
    correct_tokens = []
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i + 1].find(stress_symbol) != -1 and tokens[i + 1][-1] == ' ':
            correct_tokens.append(tokens[i] + tokens[i + 1].strip())
            correct_tokens.append(' ' * tokens[i + 1].count(' '))
            i += 2
        elif i + 2 == len(tokens) and tokens[i + 1].find(stress_symbol) != -1:
            correct_tokens.append(tokens[i] + tokens[i + 1])
            i += 2
        elif i + 2 < len(tokens) and tokens[i + 1].find(stress_symbol) != -1 and tokens[i + 1][-1] != ' ' and tokens[i][
            -1].isalpha() and \
                tokens[i + 2][0].isalpha() and i > 0:
            correct_tokens.append(tokens[i] + tokens[i + 1] + tokens[i + 2])
            i += 3
        else:
            correct_tokens.append(tokens[i])
            i += 1

    # Объединение токенов без ударения между собой. В случае некоторых ФИО это повышает точность расстановки ударений, например:
    # 'Лапшиной Ирины Александровны' - при обработке по токенам неправильное ударение в 'Ирины' ('И+рины' вместо 'Ири+ны')
    i = 0
    while i < len(correct_tokens):
        if correct_tokens[i].find(stress_symbol) == -1 and i + 1 < len(correct_tokens) and correct_tokens[i + 1].find(
                stress_symbol) == -1:
            correct_tokens[i] += correct_tokens[i + 1]
            del correct_tokens[i + 1]
        else:
            i += 1

    annotated_tokens = []
    for token in correct_tokens:
        if token.find(stress_symbol) == -1:
            annotated_tokens += list(stress_placement_obj.annotate(token))
        else:
            annotated_tokens.append(token)

    stressed_tokens = []
    for token in annotated_tokens:
        if isinstance(token, accentizer.AnnotatedToken) and token.annotation:
            stressed_tokens.append(token.annotation.variants[0].apply_to(token.string, stress_symbol,
                                                                         accentizer.StressMarkPlacement.AFTER_STRESSED_VOWEL))
        elif isinstance(token, accentizer.AnnotatedToken):
            stressed_tokens.append(token.string)
        else:
            stressed_tokens.append(token)
    stressed_text = ''.join(stressed_tokens)

    return stressed_text


def stressrnn_wrapper(text: str, stress_placement_obj: Any) -> str:
    ''' Обёртка для библиотеки stressrnn для унификации её использования в тестах. '''

    return stress_placement_obj.put_stress(text, stress_symbol='+', accuracy_threshold=0.0)


stress_rnn = StressRNN()
accentizer_from_morpher = accentizer.Accentizer(accentizer.load_standard_dictionaries())
