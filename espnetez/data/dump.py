import glob
import os
from pathlib import Path
from typing import Dict, List, Union


def join_dumps(
    dump_paths: List[str],
    dump_prefix: List[str],
    output_dir: Union[str, Path],
):
    """
    Create a joined dump file from a list of dump paths.

    This function takes multiple dump paths and prefixes, reads the corresponding
    dump files, and creates a new dump file in the specified output directory.
    Each line from the original dump files is prefixed with the corresponding
    prefix from the `dump_prefix` list.

    Args:
        dump_paths (List[str]): A list of paths for the dump directories.
                                 Each path should contain the dump files to be joined.
        dump_prefix (List[str]): A list of prefixes for the dump files.
                                 Each prefix will be added to the beginning of the
                                 corresponding lines in the joined output file.
        output_dir (Union[str, Path]): The output directory where the joined dump
                                        file will be saved. If the directory does
                                        not exist, it will be created.

    Raises:
        ValueError: If any of the expected dump files do not exist in the specified
                    dump paths.

    Examples:
        >>> join_dumps(
        ...     dump_paths=["/path/to/dump1", "/path/to/dump2"],
        ...     dump_prefix=["dataset1", "dataset2"],
        ...     output_dir="/path/to/output"
        ... )
        This will read dump files from "/path/to/dump1" and "/path/to/dump2",
        prefix the lines with "dataset1-" and "dataset2-", and write the joined
        content to "/path/to/output".

    Note:
        It is assumed that all dump directories contain the same set of dump file
        names. If the dump files have different names, a ValueError will be raised.
    """
    dump_file_names = [
        os.path.basename(g) for g in glob.glob(os.path.join(dump_paths[0], "*"))
    ]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dump_file_name in dump_file_names:
        lines = []
        for idx_dataset, dataset in enumerate(dump_paths):
            if not os.path.exists(os.path.join(dataset, dump_file_name)):
                raise ValueError(
                    f"Dump file {dump_file_name} does not exist in {dataset}."
                )

            dump_file_path = os.path.join(dataset, dump_file_name)
            with open(dump_file_path, "r") as f:
                for line in f.readlines():
                    lines.append(f"{dump_prefix[idx_dataset]}-" + line)

        with open(os.path.join(output_dir, dump_file_name), "w") as f:
            f.write("\n".join([line.replace("\n", "") for line in lines]))


def create_dump_file(
    dump_dir: Union[str, Path],
    dataset: Union[Dict[str, Dict], List[Dict]],
    data_inputs: Dict[str, Dict],
):
    """
    Create a dump file for a dataset.

    This function generates a dump file in the specified directory containing
    the specified data from the dataset. The dump file will include information
    related to the input variables as specified in the `data_inputs` argument.

    Args:
        dump_dir (Union[str, Path]):
            The output directory where the dump files will be saved.
            If the directory does not exist, it will be created.

        dataset (Union[Dict[str, Dict], List[Dict]]):
            The dataset from which to create the dump file. It can either be
            a dictionary where each key represents a data entry or a list of
            dictionaries representing multiple entries.

        data_inputs (Dict[str, Dict]):
            A dictionary containing data information for each input variable.
            Each key should correspond to a variable name, and the value should
            be a list where the first element is the desired output file name
            for that variable.

    Raises:
        ValueError:
            If `dataset` is neither a dictionary nor a list, or if any
            expected data entry is missing.

    Examples:
        Creating a dump file from a dictionary dataset:

        >>> dump_dir = "output/dump"
        >>> dataset = {
        ...     0: {"feature1": "value1", "feature2": "value2"},
        ...     1: {"feature1": "value3", "feature2": "value4"},
        ... }
        >>> data_inputs = {
        ...     "feature1": ["feature1_dump.txt"],
        ...     "feature2": ["feature2_dump.txt"],
        ... }
        >>> create_dump_file(dump_dir, dataset, data_inputs)

        This will create two files: `feature1_dump.txt` and `feature2_dump.txt`
        in the `output/dump` directory, each containing the corresponding data.

        Creating a dump file from a list dataset:

        >>> dump_dir = "output/dump"
        >>> dataset = [
        ...     {"feature1": "value1", "feature2": "value2"},
        ...     {"feature1": "value3", "feature2": "value4"},
        ... ]
        >>> data_inputs = {
        ...     "feature1": ["feature1_dump.txt"],
        ...     "feature2": ["feature2_dump.txt"],
        ... }
        >>> create_dump_file(dump_dir, dataset, data_inputs)

        Similar to the previous example, this will create the same dump files
        in the specified output directory.

    Note:
        Ensure that the output directory has the necessary write permissions
        to avoid any I/O errors during file creation.
    """
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    if isinstance(dataset, dict):
        keys = list(dataset.keys())
    elif isinstance(dataset, list):
        keys = list(range(len(dataset)))
    else:
        raise ValueError("dataset must be a dict or a list.")

    for input_name in data_inputs:
        file_path = os.path.join(dump_dir, data_inputs[input_name][0])
        text = []
        for key in keys:
            text.append(f"{key} {dataset[key][input_name]}")

        with open(file_path, "w") as f:
            f.write("\n".join(text))
    return
