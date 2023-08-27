import os
import re
import sys


def main():
    with open(filename, "r") as f:
        lines = f.read().split("\n")[:-1]
    new_lines = []
    for line in lines:
        if "\xa0" in line:
            line = line.strip("\xa0")
        if "\u200b" in line:
            line = line.replace("\u200b", "")
        if "\u200c" in line:
            line = line.replace("\u200c", "")
        skip_sentence = False
        for ch in ["#", "\\", "[", ":", "'"]:
            if ch in line:
                skip_sentence = True
        if skip_sentence:
            continue
        is_alpha = re.search("[a-zA-Z]", line)
        if is_alpha is not None:
            continue
        line = re.sub(" +", " ", line)
        new_lines.append(line)

    with open(filename, "w") as f:
        for line in new_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    filename = sys.argv[1]
    main()
