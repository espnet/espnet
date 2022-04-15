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
