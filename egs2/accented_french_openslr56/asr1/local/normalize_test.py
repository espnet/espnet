# This script is normalizing the test set :
# It removes punctuation as it is not needed for ASR task
# It also removes capital letters.
# This is not the case for the other sets (train, dev and devtest)
# Those sets are already normalized

# Example :
# Before : Oui, qu'est-ce que vous voulez?
# After : oui qu'est-ce que vous voulez


f = open("downloads/African_Accented_French/transcripts/test/ca16/prompts.txt")

new_f = open(
    "downloads/African_Accented_French/transcripts/test/ca16/new_prompts.txt", "w"
)

for row in f:
    uttid = row.split(" ")[0]
    utt = " ".join(row.split(" ")[1:])
    utt = utt.split("\n")[0]
    utt = utt.lower()
    utt = utt.strip(".?!")
    utt = utt.replace(",", "")
    utt = utt.replace(";", "")

    new_f.write(uttid + " " + utt + "\n")
