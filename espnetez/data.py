import os


def join(dataset_dict):
    # check if all datasets have the same number of samples
    first_dataset = list(dataset_dict.keys())[0]
    for key in dataset_dict.keys():
        if len(dataset_dict[key]) != len(dataset_dict[first_dataset]):
            raise ValueError("Datasets must have the same number of samples.")

    # then join all the datasets.
    dataset = {}
    first_dataset = True
    for k, d in dataset_dict.items():
        if first_dataset:
            for key in d.keys():
                dataset[key] = {k: d[key]}
            first_dataset = False
        else:
            for key in d.keys():
                dataset[key][k] = d[key]

    return dataset


def create_dump_file(dump_dir, dataset, data_inputs):
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    if isinstance(dataset, dict):
        keys = list(dataset.keys())
    elif isinstance(dataset, list):
        keys = list(range(len(dataset)))
    else:
        raise ValueError("dataset must be a dict or a list.")

    for input_name in data_inputs.keys():
        file_path = os.path.join(dump_dir, data_inputs[input_name]["file"])
        text = []
        for key in keys:
            text.append(f"{key} {dataset[key][input_name]}")

        with open(file_path, "w") as f:
            f.write("\n".join(text))
    return
