import glob
import os
from argparse import Namespace

from espnetez.utils import get_task_class
from typing import Union, Dict
from pathlib import Path

class Trainer:
    """Generic trainer class for ESPnet training!"""
    def __init__(
        self,
        task: str,
        train_dump_dir: Union[str, Path],
        valid_dump_dir: Union[str, Path],
        output_dir: Union[str, Path],
        data_inputs: Dict[str, Dict],
        train_config: Namespace,
        **kwargs
    ):
        self.task_class = get_task_class(task)
        self.train_config = train_config
        self.train_config.update(kwargs)
        self.train_config = Namespace(**self.train_config)
        self.output_dir = output_dir
        self._update_config(
            train_dump_dir=train_dump_dir,
            valid_dump_dir=valid_dump_dir,
            output_dir=output_dir,
            data_inputs=data_inputs,
            **kwargs
        )

    def _update_config(
        self, train_dump_dir, valid_dump_dir, output_dir, data_inputs, **kwargs
    ):
        train_data_path_and_name_and_type = [
            (os.path.join(train_dump_dir, df["file"]), k, df["type"])
            for k, df in data_inputs.items()
        ]
        valid_data_path_and_name_and_type = [
            (os.path.join(valid_dump_dir, df["file"]), k, df["type"])
            for k, df in data_inputs.items()
        ]
        self.train_config.train_data_path_and_name_and_type = (
            train_data_path_and_name_and_type
        )
        self.train_config.valid_data_path_and_name_and_type = (
            valid_data_path_and_name_and_type
        )
        self.train_config.output_dir = output_dir
        self.train_config.print_config = kwargs.get("print_config", False)
        self.train_config.required = kwargs.get(
            "required", ["output_dir", "token_list"]
        )

        self.train_config.train_shape_file = glob.glob(
            os.path.join(output_dir, "train", "*shape")
        )
        self.train_config.valid_shape_file = glob.glob(
            os.path.join(output_dir, "valid", "*shape")
        )

    def train(self):
        # check if output directory contains stats file.
        if len(self.train_config.train_shape_file) == 0:
            # Then we need to collect stats.
            self.train_config.collect_stats = True
            self.task_class.main(self.train_config)

        # Then start training.
        self.train_config.collect_stats = False
        self.task_class.main(self.train_config)
    
