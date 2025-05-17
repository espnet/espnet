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
