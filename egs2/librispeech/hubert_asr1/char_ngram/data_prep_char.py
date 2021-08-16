import re
import numpy
from tqdm import tqdm
def convert_words_to_letters_asg_rep2(fin_name, fout_name):
    with open(fin_name, "r") as fin, open(fout_name, "w") as fout:
        for line in tqdm(fin):
            words = line.strip().split(" ")
            for i, word in enumerate(words):
                word = re.sub("[^A-Z'.]+", "", word)
                if len(word) == 0:
                    continue
                new_word = transform_asg(word)
                fout.write(" ".join(list(new_word)))
                if i != len(words) - 1:
                    fout.write(" ")
            fout.write("\n")


def transform_asg(word):
    if word == "":
        return ""
    new_word = word[0]
    prev = word[0]
    repetition = 0
    for letter in word[1:]:
        if letter == prev:
            repetition += 1
        else:
            if repetition != 0:
                new_word += "1" if repetition == 1 else "2"
                repetition = 0
            new_word += letter
        prev = letter
    if repetition != 0:
        new_word += "1" if repetition == 1 else "2"
    return new_word
            
convert_words_to_letters_asg_rep2("data/dev/text", "char_ngram/dev_char")
