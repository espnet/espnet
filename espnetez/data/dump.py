import glob
import os
from pathlib import Path
from typing import Dict, List, Union


def join_dumps(
    dump_paths: List[str],
    output_dir: Union[str, Path],
):
    """Create a joined dump file from a list of dump paths.

    Args:
        dump_paths (List[str]): List of paths for the dump directory.
        output_dir (Union[str, Path]): Output directory of the joined dump file.

    Raises:
        ValueError: _description_
    """
    dump_file_names = [
        os.path.basename(g) for g in glob.glob(os.path.join(dump_paths[0], "*"))
    ]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dump_file_name in dump_file_names:
        lines = []
        for dataset in dump_paths:
            if not os.path.exists(os.path.join(dataset, dump_file_name)):
                raise ValueError(
                    f"Dump file {dump_file_name} does not exist in {dataset}."
                )

            dump_file_path = os.path.join(dataset, dump_file_name)
            with open(dump_file_path, "r") as f:
                lines += f.readlines()

        with open(os.path.join(output_dir, dump_file_name), "w") as f:
            f.write("\n".join([line.replace("\n", "") for line in lines]))


def create_dump_file(
    dump_dir: Union[str, Path],
    dataset: Union[Dict[str, Dict], List[Dict]],
    data_inputs: Dict[str, Dict],
):
    """Create a dump file for a dataset.

    Args:
        dump_dir (str): Output folder of the dump files.
        dataset (Union[Dict[str, Dict], List[Dict]]): Dictionary of dataset.
        data_inputs (Dict[str, List[str, str]]):
            data information for each input variables.

    """
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    if isinstance(dataset, dict):
        keys = list(dataset.keys())
    elif isinstance(dataset, list):
        keys = list(range(len(dataset)))
    else:
        raise ValueError("dataset must be a dict or a list.")

    for input_name in data_inputs.keys():
        file_path = os.path.join(dump_dir, data_inputs[input_name][0])
        text = []
        for key in keys:
            text.append(f"{key} {dataset[key][input_name]}")

        with open(file_path, "w") as f:
            f.write("\n".join(text))
    return
