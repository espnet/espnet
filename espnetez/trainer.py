import os
import glob
from argparse import Namespace
from espnetez.utils import get_task_class


class Trainer:
    def __init__(
        self,
        task,
        train_dump_dir,
        valid_dump_dir,
        output_dir,
        data_inputs,
        train_config,
        **kwargs
    ):
        self.task_class = get_task_class(task)
        self.train_config = train_config
        self.train_config.update(kwargs)
        self.train_config = Namespace(**self.train_config)
        self._update_config(
            train_dump_dir=train_dump_dir,
            valid_dump_dir=valid_dump_dir,
            output_dir=output_dir,
            data_inputs=data_inputs,
            **kwargs
        )
    
    def _update_config(self, train_dump_dir, valid_dump_dir, output_dir, data_inputs, **kwargs):
        train_data_path_and_name_and_type = [
            (os.path.join(train_dump_dir, df["file"]), k, df["type"])
            for k, df in data_inputs.items()
        ]
        valid_data_path_and_name_and_type = [
            (os.path.join(valid_dump_dir, df["file"]), k, df["type"])
            for k, df in data_inputs.items()
        ]
        self.train_config.train_data_path_and_name_and_type \
            = train_data_path_and_name_and_type
        self.train_config.valid_data_path_and_name_and_type \
            = valid_data_path_and_name_and_type
        self.train_config.output_dir = output_dir
        self.train_config.print_config = kwargs.get("print_config", False)
        self.train_config.required = kwargs.get("required", ["output_dir", "token_list"])

        self.train_config.train_shape_file = glob.glob(os.path.join(output_dir, "train", "*shape"))
        self.train_config.valid_shape_file = glob.glob(os.path.join(output_dir, "valid", "*shape"))

    def train(self):
        self.train_config.collect_stats = False
        self.task_class.main(self.train_config)
    
    def collect_stats(self):
        self.train_config.collect_stats = True
        self.task_class.main(self.train_config)
