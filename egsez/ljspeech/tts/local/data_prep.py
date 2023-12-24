import os


def get_dataset(LJS_DIRS):
    # prepare dataset
    train_dataset = {}
    test_dataset = {}
    with open(os.path.join(LJS_DIRS, "metadata.csv"), "r") as f:
        lines = f.readlines()

    for t in lines[:-50]:
        k, _, text = t.strip().split("|", 2)
        train_dataset[k] = dict(text=text)

    for t in lines[-50:]:
        k, _, text = t.strip().split("|", 2)
        test_dataset[k] = dict(text=text)

    # set speech path
    for k in train_dataset.keys():
        train_dataset[k]["speech"] = os.path.join(LJS_DIRS, "wavs", f"{k}.wav")

    for k in test_dataset.keys():
        test_dataset[k]["speech"] = os.path.join(LJS_DIRS, "wavs", f"{k}.wav")

    return train_dataset, test_dataset
