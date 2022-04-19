# This script is normalizing the test set :
# It removes punctuation as it is not needed for ASR task
# It also removes capital letters.
# This is not the case for the other sets (train, dev and devtest)
# Those sets are already normalized

# Example :
# Before : Oui, qu'est-ce que vous voulez?
# After : oui qu'est-ce que vous voulez

import argparse

parser = argparse.ArgumentParser(description="Normalize test text.")
parser.add_argument("--path_test", type=str, help="path of test text file")


def main(cmd=None):
    args = parser.parse_args(cmd)

    path = args.path_test
    f = open(path + "prompts.txt")

    new_f = open(path + "new_prompts.txt", "w")

    for row in f:
        uttid = row.split(" ")[0]
        utt = " ".join(row.split(" ")[1:])
        utt = utt.split("\n")[0]
        utt = utt.lower()
        utt = utt.strip(".?!")
        utt = utt.replace(",", "")
        utt = utt.replace(";", "")

        new_f.write(uttid + " " + utt + "\n")


if __name__ == "__main__":
    main()
