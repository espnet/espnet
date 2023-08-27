import glob
import os
from argparse import Namespace

from espnetez.utils import get_task_class


class Trainer:
    """Generic trainer class for ESPnet training!"""

    def __init__(
        self,
        task,
        train_dump_dir,
        valid_dump_dir,
        output_dir,
        data_inputs,
        stats_dir,
        train_config,
        **kwargs
    ):
        self.task_class = get_task_class(task)
        self.train_config = train_config
        self.train_config.update(kwargs)
        self.train_config = Namespace(**self.train_config)
        self.stats_dir = stats_dir
        self.output_dir = output_dir
        self._update_config(
            train_dump_dir=train_dump_dir,
            valid_dump_dir=valid_dump_dir,
            output_dir=output_dir,
            data_inputs=data_inputs,
            **kwargs
        )

    def _update_config(self, train_dump_dir, valid_dump_dir, data_inputs, **kwargs):
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
        self.train_config.print_config = kwargs.get("print_config", False)
        self.train_config.required = kwargs.get(
            "required", ["output_dir", "token_list"]
        )

    def train(self):
        # check if the stats dir exists and shape files exists.
        # if not, perform collect_stats.
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

        # check if the output dir exists and shape files exists.
        if (
            not os.path.exists(os.path.join(self.stats_dir, "train"))
            or len(glob.glob(os.path.join(self.stats_dir, "train", "*shape"))) == 0
        ):
            self.collect_stats()

        # after collect_stats, define shape files
        self.train_config.train_shape_file = glob.glob(
            os.path.join(self.stats_dir, "train", "*shape")
        )
        self.train_config.valid_shape_file = glob.glob(
            os.path.join(self.stats_dir, "valid", "*shape")
        )

        # finally start training.
        self.train_config.collect_stats = False
        self.train_config.output_dir = self.output_dir
        self.task_class.main(self.train_config)

    def collect_stats(self):
        self.train_config.collect_stats = True
        self.train_config.output_dir = self.stats_dir
        self.task_class.main(self.train_config)
