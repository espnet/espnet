import sys

error_words_freqs = {}
infile = sys.argv[1]
biasinglist = sys.argv[2]
insert_error = 0
insert_rare = 0
freqlist_test = {}

freqlist = {}
with open(biasinglist) as fin:
    rareset = set()
    for line in fin:
        rareset.add(line.strip().upper())

project_set = set()
with open(infile) as fin:
    lines = fin.readlines()
for i, line in enumerate(lines):
    if line.startswith("id:"):
        project = line.strip(")\n").split("-")[-3:]
        project = "-".join(project)
    if "REF:" in line:
        nextline = lines[i + 1].split()
        for j, word in enumerate(line.split()):
            if "*" in word:
                insert_error += 1
                if nextline[j].upper() in rareset:
                    insert_rare += 1
        line = line.replace("*", "")
        line.replace("%BCACK", "")
        for word in line.split()[1:]:
            if not word.startswith("("):
                if word.upper() not in freqlist_test:
                    freqlist_test[word.upper()] = 1
                else:
                    freqlist_test[word.upper()] += 1

                if word != word.lower() and word.upper() in error_words_freqs:
                    error_words_freqs[word.upper()] += 1
                elif word != word.lower() and word.upper() not in error_words_freqs:
                    error_words_freqs[word.upper()] = 1
                elif word == word.lower() and word.upper() not in error_words_freqs:
                    if word == word.upper():
                        print("special token found in: {}".format(project))
                    error_words_freqs[word.upper()] = 0
                elif word == word.upper():
                    print("special token found in: {}".format(project))

commonwords = []
rarewords = []
oovwords = []
common_freq = 0
rare_freq = 0
oov_freq = 0
common_error = 0
rare_error = 0
oov_error = 0
partial_error = 0
partial_freq = 0
very_common_error = 0
very_common_words = 0
words_error_freq = {}
words_total_freq = {}
low_freq_error = 0
low_freq_total = 0
for word, error in error_words_freqs.items():
    if word in rareset:
        rarewords.append(word)
        rare_freq += freqlist_test[word]
        rare_error += error
    else:
        commonwords.append(word)
        common_freq += freqlist_test[word]
        common_error += error

total_words = common_freq + rare_freq + oov_freq
insert_common = insert_error - insert_rare
total_errors = common_error + rare_error + oov_error + insert_error
WER = total_errors / total_words
print("=" * 89)
print(
    "Common words error rate: {} / {} = {}".format(
        common_error + insert_common,
        common_freq,
        round((common_error + insert_common) / common_freq, 4),
    )
)
print(
    "Rare words error rate: {} / {} = {}".format(
        rare_error + insert_rare,
        rare_freq,
        round((rare_error + insert_rare) / rare_freq, 4),
    )
)
print("WER estimate: {} / {} = {}".format(total_errors, total_words, round(WER, 4)))
print("=" * 89 + "\n")
