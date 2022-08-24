from collections import Counter

train_file = "data/train/text"

train_lines = []
with open(train_file) as f:
    for line in f:
        if not line:
            continue
        train_lines.append(line.split()[1])
train_lines = set(train_lines)

for test_name in ("test", "dev"):
    test_file = f"data/{test_name}/text"

    test_lines = []
    test_uttids = []
    with open(test_file) as f:
        for line in f:
            if not line:
                continue
            test_uttids.append(line.split()[0])
            test_lines.append(line.split()[1])

    count = 0
    duplicate_uttids = []  # duplicate ids in the test file
    for t, uttid in zip(test_lines, test_uttids):
        if t in train_lines:
            duplicate_uttids.append(uttid)
            count += 1
    duplicate_uttids = set(duplicate_uttids)
    print(count, "duplicates in", test_name)

    # if input("continue? [y/n]") == 'y':
    # remove all instances of duplicate uttids in: spk2utt, text, utt2spk, wav.scp
    with open(f"data/{test_name}/spk2utt", "r") as f:
        # replace all uttid with empty string
        text = f.read()
        for uttid in duplicate_uttids:
            text = text.replace(" " + uttid, "")
        for line in text.split("\n"):
            if not line:
                continue
            if len(line.strip().split(" ")) < 2:
                print(f"removing {line} from spk2utt")
                text = text.replace(line + "\n", "")
    with open(f"data/{test_name}/spk2utt", "w") as f:
        f.write(text)

    for name in ("text", "utt2spk", "wav.scp"):
        with open(f"data/{test_name}/{name}", "r") as f:
            # remove all lines that contain ids that correspond to duplicate sentences
            out_lines = []
            for line in f:
                if not line.split()[0] in duplicate_uttids:
                    out_lines.append(line.strip())
        with open(f"data/{test_name}/{name}", "w") as f:
            f.write("\n".join(out_lines))
            f.write("\n")
    # else:
    #     print("ok.")
