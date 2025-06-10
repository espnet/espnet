import glob
import os
from argparse import Namespace
from pathlib import Path

from espnetez.task import get_ez_task


def check_argument(
    train_dump_dir,
    valid_dump_dir,
    train_dataset,
    valid_dataset,
    train_dataloader,
    valid_dataloader,
):
    """
    Validate the arguments for training and validation data sources.

    This function checks the consistency of the input arguments used for
    specifying training and validation data sources. It ensures that
    the user adheres to the rules of specifying either dump directories,
    datasets, or dataloaders, but not a mix of them. The function will
    raise a ValueError if the conditions are not met.

    Args:
        train_dump_dir (str or None): The directory containing training
            dump files. Should be None if using datasets or dataloaders.
        valid_dump_dir (str or None): The directory containing validation
            dump files. Should be None if using datasets or dataloaders.
        train_dataset (Dataset or None): The training dataset. Should be
            None if using dump files or dataloaders.
        valid_dataset (Dataset or None): The validation dataset. Should be
            None if using dump files or dataloaders.
        train_dataloader (DataLoader or None): The training dataloader.
            Should be None if using dump files or datasets.
        valid_dataloader (DataLoader or None): The validation dataloader.
            Should be None if using dump files or datasets.

    Returns:
        bool: Returns True if all checks pass.

    Raises:
        ValueError: If any of the argument conditions are violated.

    Examples:
        # Example of valid usage with dump files
        check_argument('/path/to/train_dump', '/path/to/valid_dump', None, None, None,
            None)

        # Example of valid usage with datasets
        check_argument(None, None, train_dataset, valid_dataset, None, None)

        # Example of invalid usage - mix of dump files and datasets
        check_argument('/path/to/train_dump', None, train_dataset, None, None, None)
        # Raises ValueError: If you try to use dump file, dataset or dataloader should
            be None.

    Note:
        Ensure to specify at least one of the arguments: dump directories,
        datasets, or dataloaders. The function enforces exclusive use of
        each data source type.
    """
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
    """
    Generic trainer class for ESPnet training.

    This class is responsible for managing the training process of ESPnet models.
    It handles the configuration, dataset preparation, and the training loop.
    The Trainer class supports multiple input methods including dump directories,
    custom datasets, and dataloaders. It ensures that the provided arguments are
    consistent and valid before starting the training process.

    Attributes:
        train_config (Namespace): Configuration for training, can be a dictionary or
            Namespace object.
        task_class (Task): Task class instantiated from the provided task identifier.
        stats_dir (str): Directory where statistics for training and validation will be
            stored.
        output_dir (str): Directory where model outputs will be saved.

    Args:
        task (str): The task identifier used to retrieve the corresponding task class.
        train_config (Union[dict, Namespace]): Configuration for training.
        output_dir (str): Directory for saving model outputs.
        stats_dir (str): Directory for storing training statistics.
        data_info (dict, optional): Information about the dataset paths and types.
        train_dump_dir (str, optional): Directory containing training dump files.
        valid_dump_dir (str, optional): Directory containing validation dump files.
        train_dataset (Dataset, optional): Custom training dataset.
        valid_dataset (Dataset, optional): Custom validation dataset.
        train_dataloader (DataLoader, optional): DataLoader for training data.
        valid_dataloader (DataLoader, optional): DataLoader for validation data.
        build_model_fn (callable, optional): Function to build the model.
        **kwargs: Additional keyword arguments for configuring the training.

    Raises:
        ValueError: If any of the argument validation checks fail.

    Examples:
        >>> trainer = Trainer(
                task='asr',
                train_config={'batch_size': 32, 'learning_rate': 0.001},
                output_dir='./output',
                stats_dir='./stats',
                train_dump_dir='./train_dump',
                valid_dump_dir='./valid_dump'
            )
        >>> trainer.collect_stats()  # Collect statistics from the dataset
        >>> trainer.train()           # Start the training process

    Note:
        Ensure that either dump directories, datasets, or dataloaders are specified
        as input parameters, but not a combination of them in conflicting ways.
    """

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
            train_dump_dir = Path(train_dump_dir)
            valid_dump_dir = Path(valid_dump_dir)
            self.task_class = get_ez_task(task)
            train_dpnt = []
            valid_dpnt = []
            if "train" in data_info and "valid" in data_info:
                for k, v in data_info["train"].items():
                    train_dpnt.append((str(train_dump_dir / v[0]), k, v[1]))
                for k, v in data_info["valid"].items():
                    valid_dpnt.append((str(valid_dump_dir / v[0]), k, v[1]))
            else:
                for k, v in data_info.items():
                    train_dpnt.append((str(train_dump_dir / v[0]), k, v[1]))
                    valid_dpnt.append((str(valid_dump_dir / v[0]), k, v[1]))

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
        """
        Train the model using the specified training configuration.

        This method orchestrates the training process by first ensuring that
        the necessary shape files are available. It checks for the presence
        of shape files in the specified statistics directory, and if they
        are found, it proceeds to invoke the main training routine of the
        task class.

        Raises:
            AssertionError: If no shape files are found in the statistics
            directory for either training or validation.

        Examples:
            >>> trainer = Trainer(task='my_task', train_config=my_train_config,
                                  output_dir='output/', stats_dir='stats/')
            >>> trainer.train()  # Starts the training process
        """
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
        """
        Collects statistics for training and validation datasets.

        This method initializes the process of gathering statistical data
        from the training and validation datasets. It creates the necessary
        directories to store the statistics if they do not already exist
        and sets the configuration parameters for collecting statistics.
        The statistics are used to define the shape files required for
        training.

        The method will call the `main` function of the `task_class`
        with the updated configuration, which includes the output directory
        set to the statistics directory.

        Raises:
            OSError: If the directory for storing statistics cannot be created.

        Examples:
            >>> trainer = Trainer(task='example_task', train_config=some_config,
                                  output_dir='/path/to/output',
                                  stats_dir='/path/to/stats')
            >>> trainer.collect_stats()

        Note:
            This method must be called before training to ensure that
            the shape files are defined properly. After running this method,
            the `train_shape_file` and `valid_shape_file` attributes
            of `train_config` will be populated based on the collected
            statistics.
        """
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

        self.train_config.collect_stats = True
        self.train_config.output_dir = self.stats_dir
        self.train_config.train_shape_file = []
        self.train_config.valid_shape_file = []

        self.task_class.main(self.train_config)
