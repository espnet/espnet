import glob
import os
from argparse import Namespace

from espnetez.task import get_easy_task


class Trainer:
    """Generic trainer class for ESPnet training!"""

    def __init__(
        self,
        task,
        train_config,
        train_dump_dir,
        valid_dump_dir,
        data_info,
        output_dir,
        stats_dir,
        build_model_fn=None,
        **kwargs
    ):
        self.task_class = get_easy_task(task)
        self.train_config = train_config

        if type(self.train_config) is dict:
            self.train_config.update(kwargs)
            self.train_config = Namespace(**self.train_config)
        elif type(self.train_config) is Namespace:
            for key, value in kwargs.items():
                setattr(self.train_config, key, value)
        else:
            raise ValueError(
                "train_config should be a dict or Namespace, but got {}.".format(
                    type(self.train_config)
                )
            )

        train_dpnt = []
        valid_dpnt = []
        for k, v in data_info.items():
            train_dpnt.append((os.path.join(train_dump_dir, v[0]), k, v[1]))
            valid_dpnt.append((os.path.join(valid_dump_dir, v[0]), k, v[1]))

        self.train_config.train_data_path_and_name_and_type = train_dpnt
        self.train_config.valid_data_path_and_name_and_type = valid_dpnt

        self.stats_dir = stats_dir
        self.output_dir = output_dir
        self.train_config.print_config = kwargs.get("print_config", False)
        self.train_config.required = kwargs.get(
            "required", ["output_dir", "token_list"]
        )

        if build_model_fn is not None:
            self.task_class.build_model_fn = build_model_fn

    def train(self):
        # after collect_stats, define shape files
        self.train_config.train_shape_file = glob.glob(
            os.path.join(self.stats_dir, "train", "*shape")
        )
        self.train_config.valid_shape_file = glob.glob(
            os.path.join(self.stats_dir, "valid", "*shape")
        )
        assert (
            len(self.train_config.train_shape_file) > 0
            or len(self.train_config.valid_shape_file) > 0
        ), "You need to run collect_stats first."

        # finally start training.
        self.train_config.collect_stats = False
        self.train_config.output_dir = self.output_dir
        self.task_class.main(self.train_config)

    def collect_stats(self):
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

        self.train_config.collect_stats = True
        self.train_config.output_dir = self.stats_dir
        self.train_config.train_shape_file = []
        self.train_config.valid_shape_file = []

        self.task_class.main(self.train_config)
