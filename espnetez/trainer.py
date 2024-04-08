import glob
import os
from argparse import Namespace

from espnetez.task import get_ez_task


def check_argument(
    train_dump_dir,
    valid_dump_dir,
    train_dataset,
    valid_dataset,
    train_dataloader,
    valid_dataloader,
):
    # if we have dump files, dataset/dataloader should be None,
    # else if we have dataset, dumpfile/dataloader should be None,
    # else if we have dataloader, dumpfile/dataset should be None.
    if (train_dump_dir is not None) ^ (valid_dump_dir is not None):
        raise ValueError(
            "If you try to use dump file, both the train_dump_dir "
            + "and valid_dump_dir should be provided."
        )
    elif (
        train_dump_dir is not None
        and valid_dump_dir is not None
        and (
            train_dataset is not None
            or train_dataloader is not None
            or valid_dataset is not None
            or valid_dataloader is not None
        )
    ):
        raise ValueError(
            "If you try to use dump file, dataset or dataloader should be None."
        )

    if (train_dataset is not None) ^ (valid_dataset is not None):
        raise ValueError(
            "If you try to use custom dataset,"
            + "both the train_dataset and valid_dataset should be provided."
        )
    elif (
        train_dataset is not None
        and valid_dataset is not None
        and (train_dataloader is not None or valid_dataloader is not None)
    ):
        raise ValueError("Dataloader should be None when using custom dataset.")

    if (train_dataloader is not None) ^ (valid_dataloader is not None):
        raise ValueError(
            "If you try to use custom dataset, "
            + "both the train_dataset and valid_dataset should be provided."
        )

    if (
        train_dump_dir is None
        and valid_dump_dir is None
        and train_dataset is None
        and valid_dataset is None
        and train_dataloader is None
        and valid_dataloader is None
    ):
        raise ValueError(
            "Please specify at least one of dump_dir, dataset, or dataloader."
        )

    return True


class Trainer:
    """Generic trainer class for ESPnet training!"""

    def __init__(
        self,
        task,
        train_config,
        output_dir,
        stats_dir,
        data_info=None,
        train_dump_dir=None,
        valid_dump_dir=None,
        train_dataset=None,
        valid_dataset=None,
        train_dataloader=None,
        valid_dataloader=None,
        build_model_fn=None,
        **kwargs
    ):
        self.train_config = train_config
        check_argument(
            train_dump_dir,
            valid_dump_dir,
            train_dataset,
            valid_dataset,
            train_dataloader,
            valid_dataloader,
        )

        if type(self.train_config) is dict:
            self.train_config.update(kwargs)
            self.train_config = Namespace(**self.train_config)
        elif type(self.train_config) is Namespace:
            for key, value in kwargs.items():
                setattr(self.train_config, key, value)
        else:
            raise ValueError(
                "train_config should be dict or Namespace, but got {}.".format(
                    type(self.train_config)
                )
            )

        if train_dataset is not None and valid_dataset is not None:
            self.task_class = get_ez_task(task, use_custom_dataset=True)
            self.task_class.train_dataset = train_dataset
            self.task_class.valid_dataset = valid_dataset
        elif train_dataloader is not None and valid_dataloader is not None:
            self.task_class = get_ez_task(task, use_custom_dataset=True)
            self.task_class.train_dataloader = train_dataloader
            self.task_class.valid_dataloader = valid_dataloader
        else:
            assert data_info is not None, "data_info should be provided."
            assert train_dump_dir is not None, "Please provide train_dump_dir."
            assert valid_dump_dir is not None, "Please provide valid_dump_dir."
            self.task_class = get_ez_task(task)
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
